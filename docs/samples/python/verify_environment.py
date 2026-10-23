"""Report the imported frontend, backend, CUDA, toolchain, and GPU environment."""

import pathlib
import shutil
import subprocess

import cudnn
import torch


def first_line(command: list[str]) -> str:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return result.stdout.splitlines()[0]


def main() -> None:
    imported = pathlib.Path(cudnn.__file__).resolve()
    print(f"cudnn_file={imported}")
    print(f"frontend={cudnn.__version__}")
    print(f"backend={cudnn.backend_version_string()}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
        handle = cudnn.create_handle()
        try:
            print(f"handle_stream={cudnn.get_stream(handle)}")
        finally:
            cudnn.destroy_handle(handle)
    if shutil.which("nvcc"):
        print(f"nvcc={first_line(['nvcc', '--version'])}")
    if shutil.which("cmake"):
        print(f"cmake={first_line(['cmake', '--version'])}")
    print("environment probe ok")


if __name__ == "__main__":
    main()
