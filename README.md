# KGZSL

code for "Genre Classification Empowered by Knowledge-Embedded Music Representation"

### 1. Dependencies and Installation

```
conda create -n audio pytorch torchvision torchaudio pytorch-cuda=11.8 python=3.10 -c pytorch -c nvidia
conda activate audio
pip install lightning datasets
pip install pynvml tensorboard lz4 einops bidict
pip install scipy seaborn rich

pip install x-transformers
```

### 2. download data(fma and mtg) and run scripts in apps/data_process

### 3. Training

```
python apps/train.py \
    --input xxx/input \
    --output xxx/output \
    --data_path xxx/data \
    --config default \
    --cpu_num 0 \
    --cuda_num 1 \
    --log2file
```