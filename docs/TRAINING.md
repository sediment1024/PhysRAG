# Training

The released launcher reproduces the main two-GPU setting.

```bash
export CUDA_VISIBLE_DEVICES=0,1
export CKPT_DIR=/path/to/Wan2.2-TI2V-5B
export DATA_ROOT=$PWD/data
export OUTPUT_DIR=$PWD/output/physical_ti2v_5b_49f_bs8
export VIDEOCLIP_XL_MODEL_PATH=/path/to/VideoCLIP-XL.bin

bash finetune/scripts/build_cache_wan_49f_704_2gpu.sh
bash finetune/scripts/train_physical_ti2v_5b_2gpu_49f.sh
```

Main settings: 49 x 480 x 704, BF16, DeepSpeed ZeRO-3 with CPU offload,
learning rate 1e-6, weight decay 0.01, 20 epochs, and physical injection at
layers 0, 1, and 2.

Checkpoints are saved every 400 optimizer steps. The historical released model
uses the sparse rank-0 ZeRO-3 checkpoint format documented in the model card.
