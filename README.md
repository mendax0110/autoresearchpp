# autoresearch-cpp

C++20 / LibTorch port of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). An AI agent modifies `train` source files, builds, runs a fixed-budget experiment, and keeps changes only when val_bpb improves.

## Requirements

- CMake >= 3.25
- C++20 compiler (GCC 12+, Clang 15+, MSVC 19.34+)
- [LibTorch](https://pytorch.org/get-started/locally/) (CPU, CUDA, or MPS build)
- NVIDIA GPU optional — CPU and Apple MPS are fully supported

## Setup

### Linux + CUDA

**1. Download LibTorch**:
```bash
wget https://download.pytorch.org/libtorch/nightly/cu126/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip -d ~/libs/
```

**2. Add the NVIDIA package repository:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

**3. Install CUDA 12.1 and dependencies:**
```bash
sudo apt install --no-install-recommends \
  gcc-12 g++-12 \
  cuda-cudart-12-1 \
  cuda-compiler-12-1 \
  cuda-libraries-12-1 \
  cuda-libraries-dev-12-1 \
  libcudnn9-cuda-12 libcudnn9-dev-cuda-12 \
  libnccl2 libnccl-dev \
  libcusparselt0-cuda-12 libcusparselt0-dev-cuda-12 \
  libnvshmem3-cuda-12 libnvshmem3-dev-cuda-12
```

**4. Add CUDA to PATH** (add to `~/.bashrc`):
```bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

**5. Configure and build:**
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTorch_DIR=~/libs/libtorch/share/cmake/Torch \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
cmake --build build -j$(nproc)
```

### Linux + CPU only

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip -d ~/libs/
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTorch_DIR=~/libs/libtorch/share/cmake/Torch
cmake --build build -j$(nproc)
```

### macOS — Apple Silicon (MPS)

LibTorch with MPS support is bundled with the PyTorch pip package:

```bash
pip install torch
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTorch_DIR=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch
cmake --build build -j$(nproc)
```

### macOS — Intel (CPU only)

```bash
pip install torch
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTorch_DIR=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch
cmake --build build -j$(nproc)
```

To disable tests:
```bash
cmake -B build -DBUILD_TESTS=OFF -DTorch_DIR=...
cmake --build build -j$(nproc)
```

## Prepare data

Run copied `prepare.py` from the original repo once to download training data, then run the `convert_for_cpp.py` and produce `data/train.bin` and `data/val.bin`, then point `--data-dir` at that directory.

## Run a single experiment

```bash
# CPU
./build/autoresearchpp --device cpu --budget-secs 300

# CUDA
./build/autoresearchpp --device cuda --budget-secs 300

# Apple MPS
./build/autoresearchpp --device mps --budget-secs 300
```

```typical execution
python TensorBoardWatch.py
tensorboard --logdir runs

./autoresearchpp \
  --device cpu \
  --data-dir data \
  --budget-secs 300 \
  --seq-len 256 \
  --device-batch-size 16 \
  --total-batch-size 32768 \
  --depth 4 \
  --num-heads 4 \
  --eval-tokens 25600 \
  --eval-interval-secs 60
```


All options:
```bash
./build/autoresearchpp --help
```

## Run tests

```bash
cd build && ctest --output-on-failure
```

## Autonomous agent mode

Point your agent (Claude, Codex, etc.) at this repo and ask it to read `program.md`. The agent edits source files, builds, runs experiments, and iterates. You review `experiments.jsonl` in the morning.

## Metric

`val_bpb` (validation bits-per-byte) — lower is better, independent of vocabulary size so architectural changes compare fairly.

## License

MIT

## Credits

Data preparation (`prepare.py`) is taken from
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) (MIT License).