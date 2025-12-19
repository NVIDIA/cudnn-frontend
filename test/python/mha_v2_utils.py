import random
import math
import torch
from typing import Any, List, Optional
import cudnn

from dataclasses import dataclass, field, asdict

# Invalid left/right attention bound (negative values may be used in the future).
INVALID_BOUND = 99999


def get_strides_from_indices(
    shape, indices=[0, 1, 2, 3], gaps=[0, 0, 0, 0], rng_geom=None
):
    assert (
        len(shape) == len(gaps) == 4
        and sorted(indices) == [0, 1, 2, 3]
        and indices[3] == 3
    ), "wrong input"
    strides = [0, 0, 0, 1]  # d should always have stride 1
    curr_stride = 1

    for i in range(3, 0, -1):
        j = indices[i]

        curr_stride = (shape[j] + gaps[j]) * curr_stride
        j = indices[i - 1]
        strides[j] = curr_stride

        # Corrupt strides upwards intentionally for dim=1.
        if rng_geom is not None and shape[j] == 1:
            strides[j] = max(strides[j], rng_geom.choice([0, 3331333, 99990001]))

    total_size = shape[j] * curr_stride
    return tuple(strides), tuple(gaps), total_size


def get_strides_from_layout(shape, layout, gaps=[0, 0, 0, 0], rng_geom=None):
    assert "".join(sorted(layout)) == "bdhs", f"wrong layout '{layout}'"
    indices = ["bhsd".index(ch) for ch in layout]
    return get_strides_from_indices(shape, indices, gaps, rng_geom)


@dataclass
class exec_cfg:
    data_type: torch.dtype = None
    rng_data_seed: int = None

    is_alibi: bool = None
    is_infer: bool = True
    is_paged: bool = False
    is_bias: bool = None
    is_block_mask: bool = None
    is_padding: bool = None
    is_ragged: bool = None
    is_dropout: bool = None
    is_determin: bool = None

    diag_align: cudnn.diagonal_alignment = None
    left_bound: int = None
    right_bound: int = None

    batches: int = None
    d_qk: int = None
    d_v: int = None
    s_q: int = None
    s_kv: int = None
    h_q: int = None
    h_k: int = None
    h_v: int = None
    block_size: int = None

    # in_layout    : str = None
    # out_layout   : str = None

    shape_q: tuple[int, int, int, int] = None
    stride_q: tuple[int, int, int, int] = None
    elems_q: int = None

    shape_k: tuple[int, int, int, int] = None
    stride_k: tuple[int, int, int, int] = None
    elems_k: int = None

    shape_v: tuple[int, int, int, int] = None
    stride_v: tuple[int, int, int, int] = None
    elems_v: int = None

    shape_o: tuple[int, int, int, int] = None
    stride_o: tuple[int, int, int, int] = None
    elems_o: int = None

    seq_len_q: list[int] = field(default_factory=list)
    seq_len_kv: list[int] = field(default_factory=list)

    dropout_prob: float = 0.0

    implementation: cudnn.attention_implementation = cudnn.attention_implementation.AUTO


class RandomizationContext:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __call__(self, rng, rng_data_seed):

        randoms_ = exec_cfg()

        randoms = {}

        randoms_.rng_data_seed = rng_data_seed

        randoms["rng_data_seed"] = rng_data_seed

        self.rng_data = torch.Generator(device="cuda").manual_seed(rng_data_seed)

        randoms = {
            k: v(rng) for k, v in self.kwargs.items() if not hasattr(randoms_, k)
        }
        [
            setattr(randoms_, k, v(rng))
            for k, v in self.kwargs.items()
            if hasattr(randoms_, k)
        ]

        if "is_deterministic" in randoms:
            randoms_.is_determin = randoms["is_deterministic"] == True

        randoms_.s_q, randoms_.s_kv = randoms["s_q_s_kv"]
        randoms_.d_qk, randoms_.d_v = randoms["d_qk_d_v"]
        randoms_.h_q, randoms_.h_k, randoms_.h_v = randoms["head_count"]

        randoms_.is_ragged = randoms["is_q_ragged_or_padded_or_full"] == "ragged"
        randoms_.is_padding = (
            randoms["is_q_ragged_or_padded_or_full"] == "padded"
            or randoms["is_q_ragged_or_padded_or_full"] == "ragged"
        )

        if randoms["is_q_ragged_or_padded_or_full"] != "full":
            randoms_.seq_len_q = [
                rng.randint(1, randoms_.s_q) for _ in range(randoms_.batches)
            ]
            if randoms_.seq_len_q is not None:
                randoms_.seq_len_kv = [
                    rng.randint(randoms_.seq_len_q[i], randoms_.s_kv)
                    for i in range(randoms_.batches)
                ]
            else:
                randoms_.seq_len_kv = [
                    rng.randint(randoms_.s_q, randoms_.s_kv)
                    for _ in range(randoms_.batches)
                ]

        # Decide the left and right bounds for the sliding window mask
        randoms_.left_bound = INVALID_BOUND
        randoms_.right_bound = INVALID_BOUND

        if randoms["with_sliding_mask"] == "no_mask":
            randoms_.left_bound = INVALID_BOUND
            randoms_.right_bound = INVALID_BOUND
        elif randoms["with_sliding_mask"] == "left_window_only":
            randoms_.left_bound = rng.randint(0, randoms_.s_kv // 2)
            randoms_.right_bound = 0
        elif randoms["with_sliding_mask"] == "right_window_only":
            randoms_.left_bound = (
                INVALID_BOUND
                if randoms_.diag_align == cudnn.diagonal_alignment.BOTTOM_RIGHT
                else 1
            )
            randoms_.right_bound = rng.randint(0, randoms_.s_kv // 2)
        elif randoms["with_sliding_mask"] == "band_around_diag":
            randoms_.left_bound = rng.randint(1, randoms_.s_kv // 2)
            randoms_.right_bound = rng.randint(1, randoms_.s_kv // 2)
        elif randoms["with_sliding_mask"] == "causal":
            randoms_.right_bound = 0

        elem_align = int(16 / randoms_.data_type.itemsize)
        # Decide Q, O, Stats
        randoms_.shape_q = (randoms_.batches, randoms_.h_q, randoms_.s_q, randoms_.d_qk)
        randoms_.shape_o = (randoms_.batches, randoms_.h_q, randoms_.s_q, randoms_.d_v)

        if randoms_.is_ragged:  # Ideally Q ragged and O ragged
            randoms_.stride_q, _, randoms_.elems_q = get_strides_from_layout(
                randoms_.shape_q, "bshd"
            )
            randoms_.stride_o, _, randoms_.elems_o = get_strides_from_layout(
                randoms_.shape_o, "bshd"
            )

        else:
            indices = [0, 1, 2]
            rng.shuffle(indices)
            indices.append(3)
            gaps_q = [0, 0, 0, 0]
            gaps_o = [0, 0, 0, 0]

            if rng.randint(0, 1) == 0:  # 50% chance of gaps
                gaps_q = [rng.randint(0, 8) for _ in range(3)]
                gaps_o = [rng.randint(0, 8) for _ in range(3)]
                gaps_q.append(elem_align * rng.randint(0, 2))
                gaps_o.append(elem_align * rng.randint(0, 2))

            (randoms_.stride_q, randoms_.gaps_q, randoms_.elems_q) = (
                get_strides_from_indices(randoms_.shape_q, indices, gaps_q, rng)
            )
            (randoms_.stride_o, randoms_.gaps_o, randoms_.elems_o) = (
                get_strides_from_indices(randoms_.shape_o, indices, gaps_o, rng)
            )

        # Decide K, V
        randoms_.shape_k = (
            randoms_.batches,
            randoms_.h_k,
            randoms_.s_kv,
            randoms_.d_qk,
        )
        randoms_.shape_v = (randoms_.batches, randoms_.h_v, randoms_.s_kv, randoms_.d_v)

        if randoms_.is_ragged:  # Ideally K ragged and V ragged
            randoms_.stride_k, _, randoms_.elems_k = get_strides_from_layout(
                randoms_.shape_k, "bshd"
            )
            randoms_.stride_v, _, randoms_.elems_v = get_strides_from_layout(
                randoms_.shape_v, "bshd"
            )

        else:
            indices = [0, 1, 2]
            rng.shuffle(indices)
            indices.append(3)
            gaps_k = [0, 0, 0, 0]
            gaps_v = [0, 0, 0, 0]

            if rng.randint(0, 1) == 0:  # 50% chance of gaps
                gaps_k = [rng.randint(0, 8) for _ in range(3)]
                gaps_v = [rng.randint(0, 8) for _ in range(3)]
                gaps_k.append(elem_align * rng.randint(0, 2))
                gaps_v.append(elem_align * rng.randint(0, 2))

            (randoms_.stride_k, randoms_.gaps_k, randoms_.elems_k) = (
                get_strides_from_indices(randoms_.shape_k, indices, gaps_k, rng)
            )
            (randoms_.stride_v, randoms_.gaps_v, randoms_.elems_v) = (
                get_strides_from_indices(randoms_.shape_v, indices, gaps_v, rng)
            )

        return randoms_


class RandomChoice:
    def __init__(self, choices: dict[Any, int]):
        self.choices = choices
        self.total_weight = sum(choices.values())

    def __call__(self, rng):
        dice = rng.randint(0, self.total_weight - 1)
        for k, v in self.choices.items():
            if dice < v:
                return k
            dice -= v


class RandomIntValue:
    def __init__(
        self,
        min: int,
        max: int,
        multiple_of: Optional[int] = None,
        power_of_two: Optional[bool] = False,
        with_high_probability: Optional[List[int]] = None,
    ):
        self.min = min
        self.max = max
        self.multiple_of = multiple_of
        self.power_of_two = power_of_two
        self.with_high_probability = with_high_probability

    def __call__(self, rng):
        # 50% chance to use the with_high_probability list
        dice = rng.randint(0, 1) if self.with_high_probability is not None else 0
        if self.power_of_two:
            # calculate the smallest and largest powers of two in range, then sample
            min_exp = math.ceil(math.log2(self.min))
            max_exp = math.floor(math.log2(self.max))
            if min_exp > max_exp:
                raise ValueError(f"No power of two in range [{self.min}, {self.max}]")
            exp = (
                rng.randint(min_exp, max_exp)
                if dice == 0
                else self.with_high_probability[
                    rng.randint(0, len(self.with_high_probability) - 1)
                ]
            )
            return 1 << exp if dice == 0 else exp
        elif self.multiple_of:
            # compute the first and last valid multiples, then pick randomly
            first = (
                (self.min + self.multiple_of - 1) // self.multiple_of
            ) * self.multiple_of
            last = (self.max // self.multiple_of) * self.multiple_of
            if first > self.max:
                raise ValueError(
                    f"No multiples of {self.multiple_of} in range [{self.min}, {self.max}]"
                )
            count = ((last - first) // self.multiple_of) + 1
            idx = (
                rng.randint(0, count - 1)
                if dice == 0
                else self.with_high_probability[
                    rng.randint(0, len(self.with_high_probability) - 1)
                ]
            )
            return first + idx * self.multiple_of
        else:
            return (
                rng.randint(self.min, self.max)
                if dice == 0
                else self.with_high_probability[
                    rng.randint(0, len(self.with_high_probability) - 1)
                ]
            )


class RandomHeadGenerator:
    def __init__(self, min: int, max: int, head_group_options: tuple[int, int, int]):
        self.min = min
        self.max = max
        self.head_group_options = head_group_options

    def __call__(self, rng):
        sum_weights = sum(self.head_group_options)
        dice = rng.randint(0, sum_weights - 1)
        if dice < self.head_group_options[0]:  # MHA case
            h_q = h_k = h_v = rng.randint(self.min, self.max)
        elif dice < self.head_group_options[0] + self.head_group_options[1]:  # GQA case
            h_q = rng.randint(self.min, self.max)
            divisors = (lambda h: [i for i in range(1, h + 1) if h % i == 0])(h_q)
            group_count = rng.choice(divisors)
            h_k = h_v = h_q // group_count
        else:
            h_q = rng.randint(self.min, self.max)
            h_k = 1
            h_v = 1

        return h_q, h_k, h_v


class SlidingWindowMaskGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __call__(self, rng):
        sum_weights = sum(self.kwargs.values())
        dice = rng.randint(0, sum_weights - 1)
        for k, v in self.kwargs.items():
            if dice < v:
                return k
            dice -= v


class RandomHiddenDimSize:
    def __init__(
        self,
        d_qk_min: int,
        d_qk_max: int,
        d_v_min: int,
        d_v_max: int,
        head_dim_distribution: dict[Any, int],
        with_high_probability: Optional[List[tuple[int, int]]] = None,
    ):

        self.d_qk_gen = RandomIntValue(min=d_qk_min, max=d_qk_max, multiple_of=8)
        self.d_v_gen = RandomIntValue(min=d_v_min, max=d_v_max, multiple_of=8)
        self.distribution = RandomChoice(head_dim_distribution)
        self.with_high_probability = with_high_probability

    def __call__(self, rng):

        dice = rng.randint(0, 1) if self.with_high_probability is not None else 0

        if dice == 0:
            d_qk = self.d_qk_gen(rng)
            d_v = self.d_v_gen(rng)
            distribution = self.distribution(rng)

            if distribution == "d_qk=d_v":
                d_v = d_qk

            if d_qk < d_v:
                d_qk = d_v
        else:
            d_qk, d_v = self.with_high_probability[
                rng.randint(0, len(self.with_high_probability) - 1)
            ]

        return d_qk, d_v


class RandomSequenceLength:
    def __init__(
        self,
        s_q_min: int,
        s_q_max: int,
        s_kv_min: int,
        s_kv_max: int,
        s_q_distribution: dict[Any, int],
    ):
        self.s_q_gen = RandomIntValue(min=s_q_min, max=s_q_max)
        self.s_kv_gen = RandomIntValue(min=s_kv_min, max=s_kv_max)
        self.distribution = RandomChoice(s_q_distribution)

    def __call__(self, rng):
        s_q = self.s_q_gen(rng)
        s_kv = self.s_kv_gen(rng)

        distribution = self.distribution(rng)

        if distribution == "s_q=1":
            s_q = 1
        elif distribution == "s_q=s_kv":
            s_q = s_kv
        else:
            s_q = self.s_q_gen(rng)
            # Always s_q <=s_kv
            if s_q > s_kv:
                s_q = s_kv
        return s_q, s_kv


class RandomBatchSize(RandomIntValue):
    def __init__(
        self, min: int, max: int, with_high_probability: Optional[List[int]] = None
    ):
        super().__init__(min, max, with_high_probability=with_high_probability)

    def __call__(self, rng):
        return super().__call__(rng)


class RandomBlockSize(RandomIntValue):
    def __init__(
        self, min: int, max: int, with_high_probability: Optional[List[int]] = None
    ):
        super().__init__(
            min, max, with_high_probability=with_high_probability, power_of_two=True
        )

    def __call__(self, rng):
        return super().__call__(rng)


def test_randomization_context(seed):
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1, 4]),
        s_q_s_kv=RandomSequenceLength(
            s_q_min=1,
            s_q_max=1024,
            s_kv_min=1,
            s_kv_max=512,
            s_q_distribution={"s_q=1": 0, "s_q=s_kv": 5, "s_q=random": 10},
        ),
        d_qk_d_v=RandomHiddenDimSize(
            d_qk_min=1,
            d_qk_max=128,
            d_v_min=1,
            d_v_max=128,
            head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 1},
            with_high_probability=[(128, 128), (192, 128)],
        ),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16: 1, torch.bfloat16: 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(
            causal=10,
            left_window_only=5,
            right_window_only=5,
            band_around_diag=10,
            no_mask=10,
        ),
        diag_align=RandomChoice(
            {
                cudnn.diagonal_alignment.TOP_LEFT: 1,
                cudnn.diagonal_alignment.BOTTOM_RIGHT: 1,
            }
        ),
        is_q_ragged_or_padded_or_full=RandomChoice(
            {"ragged": 1, "padded": 1, "full": 1}
        ),
        is_kv_ragged_or_paged_or_padded_or_full=RandomChoice(
            {"ragged": 1, "paged": 1, "padded": 1, "full": 1}
        ),
        stats_layout=RandomChoice({"ragged": 1, "full": 1, "disabled": 2}),
    ) as ctx:
        return ctx


def time_execution(
    fn,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
) -> torch.Tensor:
    elapsed_times = torch.zeros(num_trials, dtype=torch.float)

    for _ in range(num_warmup):
        fn(*args)
        torch.cuda.synchronize()

    for i in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        fn(*args)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_times[i] = start_event.elapsed_time(end_event)

    return elapsed_times


def profile_execution(fn, *args, trace_dir=None):
    activities = [torch.profiler.ProfilerActivity.CUDA]
    if trace_dir:
        activities.append(torch.profiler.ProfilerActivity.CPU)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=(
            torch.profiler.tensorboard_trace_handler(trace_dir) if trace_dir else None
        ),
    ) as prof:
        fn(*args)
        torch.cuda.synchronize()

    print("Sorted by CUDA time:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    print()

    if torch.profiler.ProfilerActivity.CPU in activities:
        print("Sorted by CPU time:")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        print()


if __name__ == "__main__":
    num_tests = 10
    seed = 768
    ctx = test_randomization_context(seed=seed)
    for i in range(num_tests):
        rng = random.Random(seed + i)
        random_value = ctx(rng, seed + i)
        print(f"{i}: {random_value}")
