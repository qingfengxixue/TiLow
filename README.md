# TiLow

## 环境要求

| 组件 | 版本 | 备注 |
|------|------|------|
| 操作系统 | Ubuntu 20.04 / Windows 11 | 其他发行版未测试 |
| Python | 3.10.12 | 不兼容 3.11+ |
| GCC | 7.5.0 | 仅用于编译 OpenCV |
| CMake | 3.22.x | |
| CUDA Toolkit | 11.8 | 需匹配 PyTorch 编译版本 |
| cuDNN | 8.7.0 | |
| NVIDIA Driver | ≥ 525.60.13 | |

### 依赖安装

```bash
# 0. 安装系统级编译工具链（Linux）
sudo apt update
sudo apt install -y build-essential cmake git libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev libglfw3-dev

# 1. 先从 NVIDIA 官方归档安装 CUDA 11.8 + cuDNN 8.7
# https://developer.nvidia.com/cuda-11-8-0-download-archive
# https://developer.nvidia.com/rdp/cudnn-archive
# 安装后确保 /usr/local/cuda 指向 CUDA 11.8

# 2. 环境变量配置（写到 ~/.bashrc 或系统环境变量）
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"   # 按实际 GPU 架构修改

# 3. 创建 Python 环境（必须 3.10.12）
# conda-forge 在国内可能需要自行配置镜像源
conda create -n tilow python=3.10.12 -y
conda activate tilow

# 安装 conda-forge 的系统级图像编解码依赖
conda install -c conda-forge libjpeg-turbo=2.1.4 ffmpeg=6.1.0 -y

# 4. 安装 PyTorch Nightly (cu128)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 5. 从源码编译 OpenCV with CUDA 支持（必须，否则图像 I/O 性能不足）
git clone --branch 4.8.0 --depth 1 https://github.com/opencv/opencv.git
cd opencv
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
      -DWITH_CUDA=ON \
      -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME \
      -DWITH_CUDNN=ON \
      -DOPENCV_DNN_CUDA=ON \
      -DWITH_CUBLAS=ON \
      -DWITH_TBB=ON \
      -DBUILD_opencv_world=ON \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DWITH_FFMPEG=ON \
      -DWITH_V4L=ON \
      -DPYTHON3_EXECUTABLE=$(which python) \
      -DPYTHON3_INCLUDE_DIR=$(python -c "from sysconfig import get_paths; print(get_paths()['include'])") \
      -DPYTHON3_PACKAGES_PATH=$(python -c "import site; print(site.getsitepackages()[0])")
make -j$(nproc)
sudo make install
cd ..

# 6. 安装其余依赖（注意版本锁定，高版本可能存在 API 兼容性问题）
pip install "kornia==0.6.12" "numpy==1.24.3" "Pillow==10.0.1" "scipy==1.11.3" "pybind11==2.11.1" "tqdm==4.66.1" "pyyaml==6.0.1"
```

### 环境验证

```bash
# 必须全部返回 True 或非空才算配置正确
python -c "import torch; print(torch.cuda.is_available())"
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
python -c "import kornia; print(kornia.__version__)"
python -c "import torch; print(torch.backends.cudnn.is_available())"
```

> 如果 nightly 构建已过期，请自行调整 CUDA 版本，并确保上面四条命令全部通过。Windows 用户需自行将路径改为盘符格式（如 `D:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8`），编译 OpenCV 建议使用 Visual Studio 2019 + vcpkg，具体步骤不再赘述。

## 使用

```bash
python train.py        # 训练
python decode_test.py  # 解码
```

训练/测试数据路径与相关参数在 `train_settings.json` 和 `test_settings.json` 中配置。
