# Evaluation

Evaluation wrappers are provided for three external benchmarks:

- `evaluation/phygenbench/`
- `evaluation/videophy2/`
- `evaluation/vbench/`

Clone and install each upstream benchmark separately. Benchmark source code,
weights, prompts, generated videos, and intermediate results are not bundled.

The released defaults use 49 frames and 704 x 480 resolution. RAG is enabled
unless `--disable_rag` or `DISABLE_RAG=1` is explicitly supplied.

PhyGenBench additionally requires paths to LLaVA-NeXT, CLIP, and InternVideo2
evaluation weights through `LLAVA_MODEL_PATH`, `CLIP_MODEL_PATH`, and
`INTERNVIDEO_MODEL_PATH`.
