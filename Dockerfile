# 최신 PyTorch 컨테이너
FROM nvcr.io/nvidia/pytorch:24.07-py3

# 필수 패키지 설치
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    vim \
    libopencv-dev \
    curl \
    unzip && \
    rm -rf /var/lib/apt/lists/*

# 디렉토리 설정
WORKDIR /workspace

# pip, PyTorch + Hugging Face Transformers 설치
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install transformers datasets accelerate scipy numpy pandas tqdm matplotlib scikit-learn wandb OmegaConf python-dotenv gpustat nvitop