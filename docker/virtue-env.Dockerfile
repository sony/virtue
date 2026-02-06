## Dockerfile for virtue

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Install system requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libffi-dev \
    pkg-config \
    build-essential \
    git \
    git-lfs \
    wget

RUN pip install ninja

# customized installation
## Install VIRTUE
RUN pip install hydra-core tensorboard transformers==4.52.3 accelerate==1.7.0 datasets==3.6.0 huggingface-hub==0.33.0 numpy==1.26.4 peft==0.15.2 pillow==11.1.0 tqdm scipy==1.15.3 wrapt hjson scikit-learn==1.7.0 scikit-image==0.25.2 qwen_vl_utils ray timm==1.0.15 opencv-python==4.11.0.86 decord==0.6.0 hnswlib opencv-contrib-python==4.11.0.86 qwen-vl-utils[decord]==0.0.8


## Install SAM (Optional)
RUN git clone https://github.com/facebookresearch/sam2.git
RUN cd sam2 && \
    pip install -e . && \
    pip install -e ".[notebooks]"


## Install flash-attention
RUN wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl && pip install flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
