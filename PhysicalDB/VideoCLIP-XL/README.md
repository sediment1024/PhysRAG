---
license: cc-by-nc-sa-4.0
language:
- en
---
# What's New

[2024/10] A new [VideoCLIP-XL-v2](https://huggingface.co/alibaba-pai/VideoCLIP-XL-v2) model has been released.

[2024/10] Initial commit for the [VideoCLIP-XL](https://huggingface.co/alibaba-pai/VideoCLIP-XL) model, the [VILD](https://huggingface.co/alibaba-pai/VILD) dataset, and the [LVDR](https://huggingface.co/alibaba-pai/LVDR) benchmark.

# VideoCLIP-XL (eXtra Length)

This model is proposed from [VideoCLIP-XL paper](https://arxiv.org/abs/2410.00741). 
It aims to advance long description understanding for video CLIP Models.

# Install
~~~
# 1. Create your environment
# 2. Install torch
# 3. Then:
pip install -r requirements.txt
~~~

# Usage
Please refer to ```demo.py```.

# Source
~~~
@misc{wang2024videoclipxladvancinglongdescription,
      title={VideoCLIP-XL: Advancing Long Description Understanding for Video CLIP Models}, 
      author={Jiapeng Wang and Chengyu Wang and Kunzhe Huang and Jun Huang and Lianwen Jin},
      year={2024},
      eprint={2410.00741},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.00741}, 
}
~~~