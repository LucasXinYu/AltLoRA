# AltLoRA: Towards Better Gradient Approximation in Low-Rank Adaptation with Alternating Projections


Low-Rank Adaptation (LoRA) has emerged as an effective technique for reducing memory overhead in fine-tuning large language models, but it often suffers from sub-optimal performance compared with full fine-tuning since the update is constrained in the low-rank space. While recent variants such as LoRA-Pro attempt to mitigate this by adjusting the gradients of these low-rank matrices to better approximate the full gradient, the solution of LoRA-Pro isn't unique, and different solutions would largely influence their ablation experiment's performance. Besides, to incorporate momentum or adaptive optimization design, prior works such as LoRA-Pro need to first compute the equivalent gradient, causing a higher memory cost. It remains a mystery how to integrate momentum properly within the low-rank space for lower memory cost. In this work, we propose AltLoRA, an alternating projection method that avoids the difficulties in gradient approximation brought by the joint update design, meanwhile integrating momentum without higher memory complexity. Our theoretical analysis establishes convergence guarantees and further shows that AltLoRA enables stable feature learning and robustness to transformation invariance. Extensive experiments across multiple tasks demonstrate that AltLoRA outperforms LoRA and its variants, narrowing the gap toward full fine-tuning while preserving superior memory efficiency.

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


