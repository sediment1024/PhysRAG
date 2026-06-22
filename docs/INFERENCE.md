# Inference

Set the base model, PhysRAG checkpoint, RAG index, and VideoCLIP-XL paths:

```bash
export CKPT_DIR=/path/to/Wan2.2-TI2V-5B
export PHYSICAL_CKPT=/path/to/merged_model.pt
export FAISS_INDEX_DIR=$PWD/data/rag/faiss_index
export VIDEOCLIP_XL_MODEL_PATH=/path/to/VideoCLIP-XL.bin
```

Generate one video directly from a prompt:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune/infer_phygenbench_wan.py \
  --ckpt_dir "$CKPT_DIR" \
  --physical_ckpt "$PHYSICAL_CKPT" \
  --prompt "Molten metal is poured into a mold and cools naturally." \
  --output_dir outputs/molten_metal \
  --size "704*480" \
  --frame_num 49 \
  --faiss_index_dir "$FAISS_INDEX_DIR" \
  --videoclip_xl_model_path "$VIDEOCLIP_XL_MODEL_PATH" \
  --rag_top_k 1
```

The output is saved as `outputs/molten_metal/output.mp4`.

For PhyGenBench:

```bash
export PHYGENBENCH_ROOT=/path/to/PhyGenBench
bash finetune/scripts/infer_phygenbench_wan.sh
```

The checkpoint loader intentionally ignores empty ZeRO-3 partition tensors and
shape-mismatched base-model tensors. Use the provided loader rather than strict
`load_state_dict`.

The PhysRAG checkpoint download command will be added after the model repository
is announced.
