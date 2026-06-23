<div align="center">

# [ECCV 2026] PhysRAG

### Enhancing Physics-Awareness in Video Generation via Retrieval-Augmented Generation

Kexu Cheng · Zicheng Liu · Mingju Gao · Chunhe Song · Hao Tang

[![Code](https://img.shields.io/badge/GitHub-Code-181717?logo=github)](https://github.com/sediment1024/PhysRAG)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-PhysRAG-yellow)](https://huggingface.co/datasets/sediment1024/PhysRAG)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-PhysRAG-orange)](https://huggingface.co/sediment1024/PhysRAG)
[![Base Model](https://img.shields.io/badge/Base-Wan2.2--TI2V--5B-blue)](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
[![License](https://img.shields.io/badge/Code%20License-Apache--2.0-green)](LICENSE)

**[Paper: coming soon] · [Project Page](https://github.com/sediment1024/PhysRAG) · [Dataset](https://huggingface.co/datasets/sediment1024/PhysRAG) · [Model](https://huggingface.co/sediment1024/PhysRAG)**

</div>

<p align="center">
  <img src="assets/physrag_overview.png" width="96%" alt="PhysRAG architecture overview" />
</p>
<p align="center">
  <b>PhysRAG retrieves real-world videos exhibiting related physical dynamics,
  distills their cached VideoMAE-V2 features with learnable queries, and injects
  the resulting priors into early video diffusion transformer blocks.</b>
</p>

PhysRAG equips a pretrained text-to-video diffusion transformer with physical
priors retrieved from real-world videos. Given a text prompt, VideoCLIP-XL
retrieves a physically relevant reference from a curated database. Offline
VideoMAE-V2 features are distilled through learnable query tokens and injected
into early Wan2.2 DiT blocks, guiding generation without changing the base
text-to-video interface.

## Status

- Code: released
- Dataset: released at `sediment1024/PhysRAG`
- Model: released at `sediment1024/PhysRAG`
- Paper page: coming soon

## Quick Start

PhysRAG is easiest to reproduce in five steps:

1. Install the Python/CUDA environment from [Installation](#installation).
2. Download the four required assets: Wan2.2 base model, PhysRAG dataset,
   PhysRAG checkpoint, and VideoCLIP-XL.
3. Extract the dataset shards to a local `data/physrag/` directory.
4. Run [Inference](#inference) directly with the released checkpoint, or build
   Wan caches first if you want to reproduce [Training](#training).
5. Use the wrappers in [Evaluation](#evaluation) only after the benchmark
   repositories and evaluator weights are installed separately.

If you just want to verify the public release, start with one-prompt inference.
If you want to retrain the full method, follow the cache build plus two-GPU
training path below.

## Method

The reference encoder is run offline. During training and inference, PhysRAG
loads cached reference features, applies cross-attention with learnable queries,
and injects the resulting tokens into the denoising transformer. See
[`wan/modules/physical_adapter.py`](wan/modules/physical_adapter.py) and
[`wan/modules/physical_injection.py`](wan/modules/physical_injection.py).

## Reproduction Workflow

1. **Prepare assets**: install dependencies, then download Wan2.2, the PhysRAG
   dataset, the PhysRAG checkpoint, and VideoCLIP-XL.
2. **Extract dataset shards**: unpack the Hugging Face dataset into
   `data/physrag/` so `prompts_new.txt`, `videos_new.txt`, `videos/`, and
   `rag/` are all locally available.
3. **Build Wan caches**: precompute prompt embeddings and video latents for the
   training set with `finetune/build_cache_wan.py`.
4. **Train PhysRAG**: load cached Wan latents, retrieve physical references with
   VideoCLIP-XL + FAISS, inject physical tokens into early DiT blocks, and save
   `merged_model.pt` checkpoints.
5. **Run inference or evaluation**: load the released `merged_model.pt`, repeat
   retrieval at test time, and generate videos for prompts or benchmark suites.

## Installation

The release is tested on Linux with Python 3.10, PyTorch 2.5.1, CUDA 12.4,
FlashAttention 2.8.3, and DeepSpeed 0.18.3.

```bash
git clone https://github.com/sediment1024/PhysRAG.git
cd PhysRAG

conda create -n physrag python=3.10 -y
conda activate physrag

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -r requirements_phyrag.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

For other CUDA versions, install the corresponding PyTorch wheels. See
[`docs/INSTALL.md`](docs/INSTALL.md) and
[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

## Download Assets

```bash
# Base model
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir checkpoints/Wan2.2-TI2V-5B

# PhysRAG dataset and RAG package
huggingface-cli download sediment1024/PhysRAG \
  --repo-type dataset --local-dir data/PhysRAG

# Extract the 27 video shards and copy the RAG package
python tools/extract_dataset_shards.py \
  --dataset-dir data/PhysRAG \
  --output-dir data/physrag

# External retriever checkpoint
huggingface-cli download alibaba-pai/VideoCLIP-XL VideoCLIP-XL.bin \
  --local-dir checkpoints/VideoCLIP-XL

# PhysRAG checkpoint
huggingface-cli download sediment1024/PhysRAG \
  --repo-type model --local-dir checkpoints/PhysRAG
```

## Expected Data Layout

After extraction, the working directory used by training and inference should
look like this:

```text
data/physrag/
├── prompts_new.txt
├── videos_new.txt
├── metadata.jsonl
├── videos/<category>/<video_id>.mp4
└── rag/
    ├── metadata.jsonl
    ├── features/<category>/<video_id>.pt
    └── faiss_index/{config.json,metadata.json,video_features.index}
```

The released dataset already includes the RAG reference features and FAISS
index. The Wan prompt/video caches are not distributed and must be built
locally before training.

## Inference

Generate one 49-frame, 704×480 video with top-1 RAG retrieval:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune/infer_phygenbench_wan.py \
  --ckpt_dir checkpoints/Wan2.2-TI2V-5B \
  --physical_ckpt checkpoints/PhysRAG/merged_model.pt \
  --prompt "Molten metal is poured into a mold, flowing and cooling naturally." \
  --output_dir outputs/molten_metal \
  --size "704*480" \
  --frame_num 49 \
  --faiss_index_dir data/physrag/rag/faiss_index \
  --videoclip_xl_model_path checkpoints/VideoCLIP-XL/VideoCLIP-XL.bin \
  --rag_top_k 1
```

The generated video is written to `outputs/molten_metal/output.mp4`. For batch
PhyGenBench inference and manual reference overrides, see
[`docs/INFERENCE.md`](docs/INFERENCE.md).

The release path has been smoke-tested end to end on an NVIDIA H20 with top-1
retrieval, physical injection, one denoising step, and H.264 export at 49 frames
and 704×480 resolution.

## Training

Wan text embeddings and video latents are reproducible caches and are not
distributed. Build them before training:

The released 6,869-video dataset can be used to reproduce our training setup,
but it is not required by the method. PhysRAG can also be trained on a custom
video collection. Prepare two line-aligned files under `DATA_ROOT`: one text
prompt per line in `prompts_new.txt`, and the corresponding absolute or
`DATA_ROOT`-relative video path in `videos_new.txt`. Then point `DATA_ROOT` to
that directory and rebuild the Wan caches before training. Users may likewise
replace or extend the 170-video reference library and regenerate its
VideoMAE-V2 features and FAISS index for their own domains.

```bash
export CUDA_VISIBLE_DEVICES=0,1
export CKPT_DIR=$PWD/checkpoints/Wan2.2-TI2V-5B
export DATA_ROOT=$PWD/data/physrag
export CACHE_DIR=$DATA_ROOT/cache_wan_49f_480x704
export VIDEOCLIP_XL_MODEL_PATH=$PWD/checkpoints/VideoCLIP-XL/VideoCLIP-XL.bin

bash finetune/scripts/build_cache_wan_49f_704_2gpu.sh

export OUTPUT_DIR=$PWD/output/physrag_wan22_5b
bash finetune/scripts/train_physical_ti2v_5b_2gpu_49f.sh
```

The released configuration uses two GPUs, BF16, DeepSpeed ZeRO-3 with CPU
offload, gradient checkpointing, learning rate `1e-6`, 20 epochs, and checkpoint
saving every 400 optimizer steps. See [`docs/TRAINING.md`](docs/TRAINING.md).

## Evaluation

Wrappers are provided for:

- [PhyGenBench](evaluation/phygenbench/)
- [VideoPhy2](evaluation/videophy2/)
- [VBench](evaluation/vbench/)

Benchmark assets and evaluator checkpoints must be obtained from their original
repositories. See [`docs/EVALUATION.md`](docs/EVALUATION.md).

## License and Attribution

The PhysRAG code release is provided under Apache-2.0. Third-party code, models,
datasets, and benchmarks retain their original terms. In particular, the
VideoCLIP-XL assets are subject to their upstream license. See
[`SOURCE_LICENSE_INVENTORY.md`](SOURCE_LICENSE_INVENTORY.md) before reuse or
redistribution.

PhysRAG training videos are selected from
[`qihoo360/WISA-80K`](https://huggingface.co/datasets/qihoo360/WISA-80K).

## Acknowledgements

This project builds on [Wan2.2](https://github.com/Wan-Video/Wan2.2),
[VideoMAE-V2](https://github.com/OpenGVLab/VideoMAEv2),
[VideoCLIP-XL](https://huggingface.co/alibaba-pai/VideoCLIP-XL), and
[FAISS](https://github.com/facebookresearch/faiss). We thank the authors of
PhyGenBench, VideoPhy2, VBench, and WISA-80K for their public resources.

## Citation

```bibtex
@article{cheng2026physrag,
  title={PhysRAG: Enhancing Physics-Awareness in Video Generation via Retrieval-Augmented Generation},
  author={Cheng, Kexu and Liu, Zicheng and Gao, Mingju and Song, Chunhe and Tang, Hao},
  year={2026},
  note={Manuscript}
}
```

For details about the foundation model, refer to the official
[Wan2.2 repository](https://github.com/Wan-Video/Wan2.2).
