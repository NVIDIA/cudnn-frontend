"""Wrapper for cuDNN frontend to improve user experience.

This wrapper provides a more user-friendly interface for cuDNN frontend.
It allows users to create a graph, add operations to the graph, and then
compile the graph to a cuDNN plan. This wrapper is designed to avoid
boilerplate code.

Key Features:
    - Automatic graph validation and compilation
    - Simplified tensor management
    - Support for both named and positional tensor inputs
    - Automatic workspace management
    - PyTorch integration with DLPack support

Example:
    >>> x = torch.randn(8, 56, 56, 64, device=device, dtype=torch.float16).permute(0, 3, 1, 2)
    >>> w = torch.randn(32, 3, 3, 64, device=device, dtype=torch.float16).permute(0, 3, 1, 2)
    >>> with Graph() as graph:
    ...     y = graph.conv_fprop(
    ...         image=x, weight=w,
    ...         padding=[1,1], stride=[1,1], dilation=[1,1],
    ...         compute_data_type=data_type.FLOAT,
    ...         name="conv2d",
    ...     )
    ...     y.set_output(True).set_data_type(data_type.HALF)
    ...     # Graph is automatically validated and compiled on exit
    >>> graph.set_io_tuples(["conv2d::image", "conv2d::weight"], ["conv2d::Y"])
    >>> # Execute the graph
    >>> output = graph(x, w)
"""

from collections import OrderedDict
import atexit
import itertools
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cudnn
import cudnn.datatypes
from cudnn import data_type, heur_mode

try:
    import torch
except ImportError:
    torch = None

__all__ = ["Graph", "data_type", "heur_mode", "cudnn"]

# typedefs for readability
CudnnHandle = int
_default_cudnn_handle = None

# Configure logging
logger = logging.getLogger(__name__)


def _graph_tensor(graph: cudnn.pygraph, tensor: "torch.Tensor") -> cudnn.tensor:
    """Create a tensor in the graph object.

    Args:
        graph: The cuDNN graph object to create the tensor in
        tensor: The dlpack tensor to create a graph tensor from

    Returns:
        A cuDNN tensor object representing the input tensor in the graph

    Note:
        If the input tensor has requires_grad=True, it will be detached
        before creating the graph tensor to avoid gradient tracking issues.
    """
    if hasattr(tensor, "requires_grad") and tensor.requires_grad:
        # PyTorch tensor with requires_grad=True need to be detached first
        return graph.tensor_like(tensor.detach())
    else:
        return graph.tensor_like(tensor)


def _find_tensor(
    tensor: Union[str, cudnn.tensor, "torch.Tensor"],
    tensor_map: Dict[str, cudnn.tensor],
    dlpack_map: Dict[int, cudnn.tensor],
) -> str:
    """Find the mapping name for a tensor used in a graph.

    This function searches for a tensor in the tensor map and returns its
    corresponding name. The tensor can be specified in multiple ways:
    - As a string (either the assigned tensor name or the node::input_name)
    - As a cuDNN tensor object
    - As a DLPack-compatible tensor (e.g., PyTorch tensor) that was used in creating the graph

    Args:
        tensor: The tensor to find, can be a string name, cuDNN tensor, or DLPack tensor
        tensor_map: Dictionary mapping tensor names to cuDNN tensor objects
        dlpack_map: Dictionary mapping DLPack tensor IDs to cuDNN tensor objects

    Returns:
        The key in tensor_map that the provided tensor is mapped to

    Raises:
        ValueError: If the tensor cannot be found in the tensor map
    """
    if isinstance(tensor, str):
        # look up by canonical name, then assigned name
        if tensor in tensor_map:
            return tensor  # this name is "node::input_name"
        for tensor_name, tensor_value in tensor_map.items():
            if tensor_value.get_name() == tensor:
                return tensor_name  # name is the assigned name of the tensor
    elif isinstance(tensor, int):
        # look up by tensor uid
        for tensor_name, tensor_value in tensor_map.items():
            if tensor_value.get_uid() == tensor:
                return tensor_name
    elif isinstance(tensor, cudnn.tensor):
        for tensor_name, tensor_value in tensor_map.items():
            if tensor is tensor_value:
                return tensor_name
    elif (
        hasattr(tensor, "__dlpack__")
        and isinstance(dlpack_map, dict)
        and id(tensor) in dlpack_map
    ):
        tensor = dlpack_map[id(tensor)]
        for tensor_name, tensor_value in tensor_map.items():
            if tensor_value == tensor:
                return tensor_name
    raise ValueError("Input not found in tensor map")


def _extract_tensor(
    name: str, tensor: cudnn.tensor, arg_dict: dict
) -> Optional["torch.Tensor"]:
    """Extract a dlpack tensor from the arg_dict that matches the provided name or cudnn tensor

    Args:
        name: The name of the tensor to extract
        tensor: The cudnn tensor to extract
        arg_dict: The dictionary of arguments to extract the tensor from

    Returns:
        A dlpack tensor
    """
    if name in arg_dict:
        return arg_dict[name]  # match by canonical name
    if tensor in arg_dict:
        return arg_dict[tensor]  # match by cudnn tensor object
    try:
        return arg_dict[tensor.get_name()]  # match by assigned name
    except KeyError:
        pass
    try:
        return arg_dict[tensor.get_uid()]  # match by tensor uid
    except KeyError:
        return None  # not found


def _tensor_like(
    cudnn_tensor: cudnn.tensor, tensor_type: str = "pyt"
) -> "torch.Tensor":
    """Create a tensor like the provided cudnn tensor

    Args:
        cudnn_tensor: The cuDNN tensor to create a dlpack tensor from
        tensor_type: The type of tensor to create, currently only "pyt" is supported

    Returns:
        A dlpack tensor allocated that is like the provided cuDNN tensor
    """
    if tensor_type != "pyt":
        raise NotImplementedError("Only PyTorch tensor is supported for now")
    if not cudnn.datatypes.is_torch_available():
        raise RuntimeError("PyTorch is not available")
    dtype = cudnn.datatypes._cudnn_to_torch_data_type(cudnn_tensor.get_data_type())
    if dtype is None:
        raise TypeError(
            f"cuDNN uses an unsupported data type in PyTorch: {cudnn_tensor.get_data_type()}"
        )
    tensor = torch.empty(cudnn_tensor.get_dim(), device="cuda", dtype=dtype)
    tensor = torch.as_strided(tensor, cudnn_tensor.get_dim(), cudnn_tensor.get_stride())
    return tensor


def get_default_handle(stream: Optional["torch.cuda.Stream"] = None) -> CudnnHandle:
    """Get the default cuDNN handle and set to torch's current stream"""
    global _default_cudnn_handle
    if torch is None:
        raise RuntimeError("PyTorch is not available")
    if _default_cudnn_handle is None:
        _default_cudnn_handle = cudnn.create_handle()
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=_default_cudnn_handle, stream=stream)
    return _default_cudnn_handle


def destroy_default_handle():
    if _default_cudnn_handle is not None:
        cudnn.destroy_handle(_default_cudnn_handle)


atexit.register(destroy_default_handle)


class Graph:
    """Wrapper object for cuDNN computation graph

    This class simplifies the process of creating, compiling, and executing
    cuDNN computation graphs. It handles common boilerplate code and provides
    a more Pythonic interface to the cuDNN frontend API.

    Key features:
    - Automatic graph validation and compilation
    - Simplified tensor management with PyTorch integration
    - Support for both named and positional tensor inputs
    - Automatic workspace management

    Note:
        The graph is automatically validated and compiled when exiting the
        context manager. Any errors in graph construction will be raised
        at that point.
    """

    __handle: Optional[CudnnHandle] = None  # holding the cudnn handle pointer

    def __init__(
        self,
        *,
        handle: Optional[CudnnHandle] = None,
        inputs: Optional[List[Union[str, "torch.Tensor", cudnn.tensor]]] = None,
        outputs: Optional[List[Union[str, "torch.Tensor", cudnn.tensor]]] = None,
        heuristics: Optional[List[heur_mode]] = None,
        workspace_alloc: bool = True,
        **kwargs,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not available")
        if inputs and not isinstance(inputs, (list, tuple)):
            raise ValueError("inputs must be a list or tuple")
        if outputs and not isinstance(outputs, (list, tuple)):
            raise ValueError("outputs must be a list or tuple")
        if heuristics and not isinstance(heuristics, (list, tuple)):
            raise ValueError("heuristics must be a list or tuple")
        if cudnn.backend_version() < 91200:
            raise RuntimeError("cuDNN version 9.12.0 or higher is required")
        self.__kwargs = kwargs
        self.__graph = None  # to hold the cudnn.pygraph object
        self.__tensor_map = {}  # obj id of dlpack tensor -> cudnn tensor
        self.__tensor_in = (
            OrderedDict()
        )  # canonical node::argname -> cudnn tensors used as the input
        self.__tensor_out = (
            OrderedDict()
        )  # canonical node::outname -> cudnn tensors produced by the node
        self.__tensor_unknown = []  # list of cuDNN tensors created by user directly
        self.__node_count = {}  # function name of graph node -> number of times used
        self.__node_names = (
            set()
        )  # set of assigned names of graph nodes, to check name collision
        self.__input_tuples = None  # tuple of input tensors, if set by set_io_tuples
        self.__output_tuples = None  # tuple of output tensors, if set by set_io_tuples
        self.__inputs = (
            inputs or []
        )  # hold the list of inputs, to be used by set_io_tuples() implicitly
        self.__outputs = (
            outputs or []
        )  # hold the list of outputs, to be used by set_io_tuples() implicitly
        self.__heuristics = heuristics or [heur_mode.A, heur_mode.FALLBACK]
        if not workspace_alloc:
            self.__workspace = False
        if handle:
            self.__handle = handle
        # silently replace the PyTorch dtype into cuDNN dtype
        for key in ["io_data_type", "intermediate_data_type", "compute_data_type"]:
            if key in kwargs:
                kwargs[key] = (
                    cudnn.datatypes._torch_to_cudnn_data_type(kwargs[key])
                    or kwargs[key]
                )

    def __del__(self):
        pass

    def __enter__(self):
        if self.__graph is not None:
            raise RuntimeError("Graph already created")
        self.__graph = cudnn.pygraph(
            # Pass handle only if self.__handle is not None
            **(
                {"handle": self.__handle} if self.__handle not in ["auto", None] else {}
            ),
            **self.__kwargs,
        )
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """Exit the context manager, validating and compiling the graph.

        This method performs the following steps:
        1. Validates the graph structure
        2. Builds the operation graph
        3. Creates execution plans
        4. Checks hardware support
        5. Builds the final plans
        6. Allocates workspace memory

        Raises:
            ValidationError: If graph validation fails
            GraphStateError: If graph operations are performed in invalid order
            CudnnError: For other cuDNN-related errors
        """
        # if there is an exception, clean up and propagate
        if exc_type is not None:
            logger.error("Exception during graph construction: %s", exc_value)
            self.__graph = None
            raise
        # prepare the graph and build plans: Each should return None or raise exception
        self.__graph.validate()
        self.__graph.build_operation_graph()
        self.__graph.create_execution_plans(self.__heuristics)
        # TODO: let user select_behavior_notes() and select_numeric_notes() here
        self.__graph.check_support()
        self.__graph.build_plans()
        # Set up workspace if not forbidden by user, then set up I/O tensor orders
        if not hasattr(self, "__workspace"):
            self.__workspace = torch.empty(
                self.__graph.get_workspace_size(),
                device="cuda",
                dtype=torch.uint8,
            )
        if self.__inputs or self.__outputs:
            self.set_io_tuples(self.__inputs, self.__outputs)
        del self.__inputs, self.__outputs

        logger.debug("Inputs: %s", self.__tensor_in)
        logger.debug("Outputs: %s", self.__tensor_out)
        logger.debug("Node count: %s", self.__node_count)
        return self.__graph

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.__graph, name)
        # calling tensor_like is unnecessary, just pass through
        pass_through = [
            "get_workspace_size",
            "get_workspace_size_plan_at_index",
            "serialize",
            "deserialize",
            "query_tensor_attributes_of_uid",
        ]
        if name in pass_through:
            return attr
        # some methods are blocked and should not be called via wrapper
        # TODO should allow user select execution plan
        blocked_methods = [
            "build",
            "build_operation_graph",
            "build_plan_at_index",
            "build_plans",
            "check_support",
            "create_execution_plan",
            "create_execution_plans",
            "deselect_behavior_notes",
            "deselect_engines",
            "deselect_numeric_notes",
            "deselect_workspace_greater_than",
            "execute",
            "execute_plan_at_index",
            "get_behavior_notes",
            "get_behavior_notes_for_plan_at_index",
            "get_engine_count",
            "get_execution_plan_count",
            "get_knobs_for_engine",
            "get_plan_name_at_index",
            "key",
            "populate_cuda_graph",
            "query_tensor_attributes_of_uid",
            "select_behavior_notes",
            "select_numeric_notes",
            "update_cuda_graph",
            "validate",
        ]

        if name in blocked_methods:
            raise RuntimeError(f"Calling {name} via wrapper is not allowed")
        # non-methods: pass through. Probably not used but be safe
        if not inspect.ismethod(attr):
            return attr

        # tensor creation methods: capture the output
        def tensor_capture(*args, **kwargs):
            output = attr(*args, **kwargs)
            self.__tensor_unknown.append(output)
            return output

        if name in ["tensor", "tensor_like"]:
            return tensor_capture

        # other methods: wrap the method to intercept the arguments and return values
        def wrapper(*args, **kwargs):
            args = list(args)  # shallow copy, to allow in-place modification
            # determine the name of the graph node, the node may carry name attribute
            if name not in self.__node_count:
                self.__node_count[name] = 0
            self.__node_count[name] += 1
            if "name" in kwargs:
                node_name = kwargs["name"]
            else:
                node_name = f"{name}.{self.__node_count[name]-1}"
                kwargs["name"] = node_name
            if node_name in self.__node_names:
                raise ValueError(f"Node name {node_name} already used")
            self.__node_names.add(node_name)
            # process positional arguments for dlpack tensors
            for i, obj in enumerate(args):
                if hasattr(obj, "__dlpack__"):
                    obj_id = id(obj)
                    if obj_id not in self.__tensor_map:
                        self.__tensor_map[obj_id] = _graph_tensor(self.__graph, obj)
                    obj = args[i] = self.__tensor_map[obj_id]
                if isinstance(obj, cudnn.tensor):
                    self.__tensor_in[f"{node_name}::{i}"] = obj
            # process keyword arguments for dlpack tensors
            for key, obj in kwargs.items():
                if hasattr(obj, "__dlpack__"):
                    obj_id = id(obj)
                    if obj_id not in self.__tensor_map:
                        self.__tensor_map[obj_id] = _graph_tensor(self.__graph, obj)
                    obj = kwargs[key] = self.__tensor_map[obj_id]
                if isinstance(obj, cudnn.tensor):
                    self.__tensor_in[f"{node_name}::{key}"] = obj
            # capturing node output
            output = attr(*args, **kwargs)
            if isinstance(output, cudnn.tensor):
                output_list = [output]
            elif isinstance(output, (list, tuple)):
                output_list = output
            for i, obj in enumerate(output_list):
                if isinstance(obj, cudnn.tensor):
                    if hasattr(obj, "get_name") and obj.get_name():
                        tensor_name = obj.get_name()
                    else:
                        tensor_name = f"{node_name}::{i}"
                    self.__tensor_out[tensor_name] = obj
            return output

        return wrapper

    def __call__(self, *args, **kwargs):
        """Execute the graph with tensor dict"""
        if self.__graph is None:
            raise RuntimeError("Graph not created")
        if not self.__graph.get_execution_plan_count():
            raise RuntimeError(
                "You should not invoke the graph before the context exits"
            )
        if len(args) == 1 and isinstance(args[0], dict):
            return self.__call_with_tensor_dict(args[0], **kwargs)
        else:
            if len(args) > 0 and not self.__input_tuples:
                raise ValueError(
                    "You should not invoke the graph with positional arguments before running set_io_tuples()"
                )
            if len(args) != len(self.__input_tuples):
                raise ValueError(
                    f"Number of arguments ({len(args)}) does not match number of inputs ({len(self.__input_tuples)})"
                )
            return self.__call_with_positional_args(*args, **kwargs)

    def __call_with_positional_args(
        self, *args, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", ...]]:
        """Execute the graph with positional arguments.

        Args:
            *args: Positional arguments to pass to the graph
            **kwargs: Additional keyword arguments to pass to the graph execution

        Returns:
            A single tensor or a tuple of tensors

        Note:
            This method is called by __call__() when the graph is executed with positional arguments.
            It is not intended to be called directly by the user. The `args` should be a list of dlpack tensors
            that matches the input order of `self.__input_tuples`.
        """
        # prepare the variant pack:
        # all non-virtual tensors in __tensor_in and __tensor_out should be filled
        variant_pack = {}
        for cudnn_tensor, user_tensor in zip(self.__input_tuples, args):
            variant_pack[cudnn_tensor.get_uid()] = user_tensor
        output_tuple = [
            _tensor_like(cudnn_tensor, "pyt") for cudnn_tensor in self.__output_tuples
        ]
        for cudnn_tensor, user_tensor in zip(self.__output_tuples, output_tuple):
            variant_pack[cudnn_tensor.get_uid()] = user_tensor
        # execute the graph
        kwargs = dict(kwargs)  # shallow copy

        if "handle" not in kwargs:
            if self.__handle == "auto":
                kwargs["handle"] = get_default_handle()
            elif self.__handle is not None:
                kwargs["handle"] = self.__handle
            else:
                raise RuntimeError("Need to specify cudnn handle to execute graph")
        if "workspace" not in kwargs:
            if self.__workspace is False:
                raise RuntimeError("Need to specify workspace to execute graph")
            else:
                kwargs["workspace"] = self.__workspace
        self.__graph.execute(variant_pack, **kwargs)
        # return the output as a single tensor or a tuple
        if len(output_tuple) == 1:
            return output_tuple[0]
        else:
            return output_tuple

    def __call_with_tensor_dict(
        self,
        tensor_dict: Dict[str, "torch.Tensor"],
        **kwargs,
    ) -> Dict[str, "torch.Tensor"]:
        """Execute the graph with a dictionary of tensors.

        Args:
            tensor_dict: Dictionary of tensor names to tensors
            **kwargs: Additional keyword arguments to pass to the graph execution

        Returns:
            Dictionary of tensor names to tensors

        Raises:
            RuntimeError: If a non-virtual tensor in the graph is not found in
            `tensor_dict`, or the tensor in `tensor_dict` is not a dlpack tensor
        """
        # Notes
        """
        from arg tensor_dict -> variant_pack
        also from self.__tensor_in + self.__tensor_out -> variant_pack
        both: check if all non-virtual tensors are filled
        """
        # prepare the variant pack:
        # all non-virtual tensors in __tensor_in and __tensor_out should be filled
        variant_pack = {}
        missing_tensors = {}
        for name, tensor in itertools.chain(
            self.__tensor_in.items(), self.__tensor_out.items()
        ):
            if tensor.get_uid() in variant_pack or tensor.get_is_virtual():
                continue  # already filled or not needed
            user_tensor = _extract_tensor(name, tensor, tensor_dict)
            if user_tensor is None:
                missing_tensors[tensor] = name  # overwriting existing entries
                continue
            if not hasattr(user_tensor, "__dlpack__"):
                raise RuntimeError(f"Tensor {name} is not provided as a dlpack tensor")
            variant_pack[tensor.get_uid()] = user_tensor
        # check if all non-virtual tensors are filled
        missing_inputs = []
        missing_outputs = []
        for tensor, name in missing_tensors.items():
            if tensor.get_uid() in variant_pack:
                continue  # already filled
            if name in self.__tensor_out:
                # output tensor not specified, should be created automatically
                variant_pack[tensor.get_uid()] = tensor_dict[name] = _tensor_like(
                    tensor, "pyt"
                )
                missing_outputs.append(name)
            else:
                # input tensor not specified, flag it as missing
                missing_inputs.append(name)
        if missing_inputs:
            raise RuntimeError(
                f"Non-virtual input tensors not found in variant pack: {missing_inputs}"
            )
        if missing_outputs:
            logger.debug("Added output tensors: %s", missing_outputs)
        # execute the graph
        kwargs = dict(kwargs)  # shallow copy
        if "handle" not in kwargs:
            if self.__handle == "auto":
                kwargs["handle"] = get_default_handle()
            elif self.__handle is not None:
                kwargs["handle"] = self.__handle
            else:
                raise RuntimeError("Need to specify cudnn handle to execute graph")
        if "workspace" not in kwargs:
            if self.__workspace is False:
                raise RuntimeError("Need to specify workspace to execute graph")
            else:
                kwargs["workspace"] = self.__workspace
        self.__graph.execute(variant_pack, **kwargs)
        return tensor_dict  # by this time, the output tensors are updated

    def set_io_tuples(
        self,
        inputs: List[Union[str, "torch.Tensor", cudnn.tensor]],
        outputs: List[Union[str, "torch.Tensor", cudnn.tensor]],
    ) -> None:
        """Set order of input and output tensors to allow graph to be executed with positional arguments.

        Args:
            inputs: List of input tensors or names
            outputs: List of output tensors or names

        Raises:
            ValueError: If inputs or outputs are not lists or tuples
        """
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("inputs must be a list or tuple")
        if not isinstance(outputs, (list, tuple)):
            raise ValueError("outputs must be a list or tuple")
        if not self.__graph.get_execution_plan_count():
            # raise RuntimeError("You should not invoke set_io_tuples() before the context exits")
            self.__inputs = inputs
            self.__outputs = outputs
            return

        # self.__tensor_out and self.__tensor_in are dict of str -> cudnn tensor
        # all non-virtual tensors should be either input or output

        # Convert "inputs" to a list of names that can be looked up in __tensor_in
        input_tensors = []
        tensors_found = set()
        for i, name in enumerate(inputs):
            try:
                if name in self.__tensor_unknown:
                    input_tensors.append(name)  # user-created cuDNN tensor object
                    continue
                name = _find_tensor(name, self.__tensor_in, self.__tensor_map)
                tensor = self.__tensor_in[name]
                if id(tensor) in tensors_found:
                    raise ValueError(f"Input at index {i} ({name}) is a duplicate")
                tensors_found.add(id(tensor))
                input_tensors.append(tensor)
            except ValueError:
                raise ValueError(
                    f"Input at index {i} ({name}) not found in tensor map"
                ) from None
        # Convert "outputs" to a list of names that can be looked up in __tensor_out
        output_tensors = []
        for i, name in enumerate(outputs):
            try:
                if name in self.__tensor_unknown:
                    output_tensors.append(name)  # user-created cuDNN tensor object
                    continue
                name = _find_tensor(name, self.__tensor_out, self.__tensor_map)
                tensor = self.__tensor_out[name]
                if id(tensor) in tensors_found:
                    raise ValueError(f"Output at index {i} ({name}) is a duplicate")
                tensors_found.add(id(tensor))
                output_tensors.append(tensor)
            except ValueError:
                raise ValueError(
                    f"Output at index {i} ({name}) not found in tensor map"
                ) from None
        # Verify that all input tensors are non-virtual
        for i, tensor in enumerate(input_tensors):
            if tensor.get_is_virtual():
                raise ValueError(f"Input at index {i} is a virtual tensor")
        # Verify that all non-virtual tensors are covered by input or output
        for name, tensor in self.__tensor_out.items():
            if not tensor.get_is_virtual() and tensor not in output_tensors:
                raise ValueError(
                    f"Node output {name} is a non-virtual tensor but not specified as output"
                )
        for name, tensor in self.__tensor_in.items():
            if not tensor.get_is_virtual() and id(tensor) not in tensors_found:
                raise ValueError(
                    f"Node input {name} is a non-virtual tensor but not specified as input or output"
                )
        # Set the input and output names
        self.__input_tuples = tuple(input_tensors)
        self.__output_tuples = tuple(output_tensors)
