# AltLoRA: Optimizing Low-Rank Adaptation with Alternating Projection and Momentum Alignment


Low-Rank Adaptation (LoRA) has emerged as an effective technique for reducing memory overhead in fine-tuning large language models, but it often suffers from sub-optimal performance. While recent variants such as LoRA-Pro attempt to mitigate this through improved gradient approximation, two key challenges remain unresolved: overlapping low-rank subspaces and momentum misalignment. These issues impede optimal gradient recovery in practice and lead to degraded performance especially when momentum is used. In this work, we propose AltLoRA, an alternating projection method that separates the updates of the low-rank components, ensuring efficient full gradient approximation and consistent momentum alignment within low-rank spaces. Without allowing full-parameter learning, we provide a parameter-efficient method to optimize gradient and momentum properly and dynamically within the low-rank spaces. Our theoretical analysis establishes convergence guarantees and further shows that AltLoRA enables stable feature learning and robustness to transformation invariance. Extensive experiments across multiple tasks demonstrate that AltLoRA outperforms LoRA and its variants, narrowing the gap toward full fine-tuning while preserving superior memory efficiency.

---

## Requirements

### Installation

Create a conda environment and install dependencies:

```bash
[git clone https://github.com/[your-username]/[your-repo-name].git](https://anonymous.4open.science/r/AltLoRA-DB7C/)
cd AltLoRA-DB7C

conda create -n altlora  python=3.9
conda activate altlora

# install required packages
pip install -r requirements.txt
pip install -e peft


