# inpaint_asuka

## Create env

First create conda env python 3.11

```bash
conda create -n asuka python=3.11
conda activate asuka
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Prepare checkpoints

Download the MAE checkpoint from hf by running the following command:

```bash
mkdir checkpoints
wget https://huggingface.co/inpaint-context/mae/resolve/main/checkpoint-2690-v0.zip -O checkpoints/checkpoint-2690-v0.zip
unzip checkpoints/checkpoint-2690-v0.zip -d checkpoints
rm checkpoints/checkpoint-2690-v0.zip
```
