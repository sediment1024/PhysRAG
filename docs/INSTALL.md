# Installation

## Requirements

- Linux
- Python 3.10
- CUDA-compatible PyTorch
- FFmpeg
- Two GPUs for the released training configuration

```bash
git clone https://github.com/spoil1024/PhysRAG.git
cd PhysRAG
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -r requirements_phyrag.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

The exact locally tested package versions are listed in
[`requirements-tested.txt`](../requirements-tested.txt). Select a PyTorch wheel
appropriate for your CUDA installation rather than copying the CUDA 12.4 command
unchanged on every machine.

Download `Wan-AI/Wan2.2-TI2V-5B` separately and set `CKPT_DIR`. Download
VideoCLIP-XL separately to `PhysicalDB/VideoCLIP-XL/VideoCLIP-XL.bin`, or set
`VIDEOCLIP_XL_MODEL_PATH`.

Download the PhysRAG dataset from `sediment1024/PhysRAG`, then extract its video
shards as described in [DATA.md](DATA.md).
