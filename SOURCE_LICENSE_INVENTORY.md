# Source and license inventory

This inventory records the provenance of release assets. It is not legal advice.

| Component | Source | Role | Distributed here | License/status |
|---|---|---|---|---|
| Wan2.2 code | `Wan-Video/Wan2.2` | Base generation code | Yes, modified | Apache-2.0 |
| Wan2.2 TI2V-5B weights | `Wan-AI/Wan2.2-TI2V-5B` | Base model | No | Apache-2.0 per local model card |
| WISA-80K | [`qihoo360/WISA-80K`](https://huggingface.co/datasets/qihoo360/WISA-80K), revision `dddbd5683581c2ebf0b463e2b1c3342b2094bfb3` | Source videos and metadata | Selected videos in dataset repo | Apache-2.0 on the official dataset card; cite the dataset and WISA paper |
| VideoMAE V2 | `OpenGVLab/VideoMAEv2` | Offline reference feature extraction | Utility integration only; no weights | MIT for source code |
| VideoCLIP-XL | `alibaba-pai/VideoCLIP-XL` | Text-video retrieval | Minimal inference source and tokenizer vocabulary; no weights | CC-BY-NC-SA-4.0 per upstream model card |
| Qwen3-VL | official Qwen repository/model | Data filtering | No code or weights copied | External dependency; record exact model revision before release |
| OpenAI CLIP | official CLIP model | Caption filtering | No weights copied | External dependency; record exact model revision before release |
| FAISS | Meta FAISS | Similarity index | Serialized index only | MIT for FAISS source |
| PhyGenBench | upstream benchmark | Evaluation | Wrapper only | Upstream license not found in local checkout; do not copy benchmark assets |
| VideoPhy2 | upstream benchmark | Evaluation | Wrapper only | Upstream license not found in local checkout; do not copy benchmark assets |
| VBench | upstream benchmark | Evaluation | Wrapper only | Upstream terms apply; do not copy benchmark assets |

## Dataset release notes

- The 6,869 videos are selected from WISA-80K and retain upstream provenance.
- The 170 RAG references are a marked subset and are not duplicated.
- Prompts are the captions used by this project.
- The official dataset card reports 79,480 rows and identifies the source paper
  as *WISA: World Simulator Assistant for Physics-Aware Text-to-Video
  Generation* (arXiv:2503.08153; NeurIPS 2025).

Canonical citation supplied by the official WISA-80K dataset card:

```bibtex
@article{wang2025wisa,
  title={WISA: World Simulator Assistant for Physics-Aware Text-to-Video Generation},
  author={Wang, Jing and Ma, Ao and Cao, Ke and Zheng, Jun and Zhang, Zhanjie and Feng, Jiasong and Liu, Shanyuan and Ma, Yuhang and Cheng, Bo and Leng, Dawei and Yin, Yuhui and Liang, Xiaodan},
  journal={arXiv preprint arXiv:2503.08153},
  year={2025}
}
```

## Model release notes

- The base Wan2.2 weights are not redistributed.
- VideoMAE-V2 and VideoCLIP-XL weights are not redistributed.
- The PhysRAG model repository will be announced separately. No PhysRAG model
  weights are included in this code repository.
