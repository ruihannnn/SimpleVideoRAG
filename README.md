
## 环境配置
> 主要改动：**去除了videorag中的知识图谱部分**

- **配置videorag环境**
```shell
# Create and activate conda environment
conda create --name videorag python=3.11
conda activate videorag
```
```shell
# Core numerical and deep learning libraries
pip install numpy==1.26.4
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
pip install accelerate==0.30.1
pip install bitsandbytes==0.43.1

# Video processing utilities
pip install moviepy==1.0.3
pip install git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
pip install --no-deps git+https://github.com/facebookresearch/ImageBind.git@3fcf5c9039de97f6ff5528ee4a9dce903c5979b3

# Multi-modal and vision libraries
pip install timm ftfy regex einops fvcore eva-decord==0.6.1 iopath matplotlib types-regex cartopy

# Audio processing and vector databases
pip install ctranslate2==4.4.0 faster_whisper==1.0.3 neo4j hnswlib xxhash nano-vectordb

# Language models and utilities
pip install transformers==4.37.1
pip install tiktoken openai tenacity
pip install ollama==0.5.3
```
```shell
# Download MiniCPM-V model
git lfs clone https://huggingface.co/openbmb/MiniCPM-V-2_6-int4

# Download Whisper model
git lfs clone https://huggingface.co/Systran/faster-distil-whisper-large-v3

# Download ImageBind checkpoint

# ！！！cd到项目文件夹下！！！
mkdir .checkpoints
cd .checkpoints
wget https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
cd ../
```


## Quick Start

**需要修改sh脚本中的环境变量**

```shell
conda activate videorag
bash quickstart.sh
```

