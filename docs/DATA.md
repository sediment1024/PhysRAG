# Dataset preparation

The Hugging Face dataset `sediment1024/PhysRAG` contains 6,869 videos packaged
as tar shards, metadata, 170 cached reference features, and a FAISS index.

Expected extracted layout:

```text
data/
  prompts_new.txt
  videos_new.txt
  metadata.jsonl
  videos/<category>/<video_id>.mp4
  rag/
    metadata.jsonl
    features/<category>/<video_id>.pt
    faiss_index/{config.json,metadata.json,video_features.index}
```

Extract shards:

```bash
python tools/extract_dataset_shards.py \
  --dataset-dir /path/to/PhysRAG-dataset \
  --output-dir ./data
```

Wan latent/text caches are intentionally not distributed. Build them using
`finetune/scripts/build_cache_wan_49f_704_2gpu.sh`.
