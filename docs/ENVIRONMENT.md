# Tested environment

The local release was smoke-tested with the following environment:

- Ubuntu Linux
- Python 3.10
- NVIDIA H20
- CUDA 12.4 runtime through PyTorch
- PyTorch 2.5.1
- FlashAttention 2.8.3
- DeepSpeed 0.18.3

The complete package snapshot is recorded in `requirements-tested.txt`.
For a fresh CUDA 12.4 environment:

```bash
conda create -n physrag python=3.10 -y
conda activate physrag

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -r requirements_phyrag.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

`requirements-tested.txt` is an environment record, not a universal lock file:
CUDA-specific PyTorch wheels must be selected for the host driver/toolkit.
