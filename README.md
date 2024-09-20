# inpaint_asuka

## Create env

First create conda env python 3.11

```bash
conda create -n asuka python=3.11
conda activate asuka
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Prepare checkpoints

Download the MAE checkpoint from hf by running the following command:

```bash
mkdir weights
wget https://huggingface.co/inpaint-context/mae/resolve/main/checkpoint-2690-v0.zip -O weights/checkpoint-2690-v0.zip
unzip weights/checkpoint-2690-v0.zip -d weights
rm weights/checkpoint-2690-v0.zip
```
