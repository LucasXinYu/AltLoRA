# AltLoRA: Towards Better Gradient Approximation in Low-Rank Adaptation with Alternating Projections


#### Low-Rank Adaptation (LoRA) has emerged as an effective technique for reducing memory overhead in fine-tuning large language models. However, it often suffers from sub-optimal performance compared with full fine-tuning since the update is constrained in the low-rank space. Recent variants such as LoRA-Pro attempt to mitigate this by adjusting the gradients of the low-rank matrices to approximate the full gradient. However, LoRA-Pro's solution is not unique, and different solutions can lead to significantly varying performance in ablation studies. Besides, to incorporate momentum or adaptive optimization design, approaches like LoRA-Pro must first compute the equivalent gradient, causing a higher memory cost close to full fine-tuning. A key challenge remains in integrating momentum properly into the low-rank space with lower memory cost. In this work, we propose AltLoRA, an alternating projection method that avoids the difficulties in gradient approximation brought by the joint update design, meanwhile integrating momentum without higher memory complexity. Our theoretical analysis provides convergence guarantees and further shows that AltLoRA enables stable feature learning and robustness to transformation invariance. Extensive experiments across multiple tasks demonstrate that AltLoRA outperforms LoRA and its variants, narrowing the gap toward full fine-tuning while preserving superior memory efficiency.
---

## Requirements

### Installation

Create a conda environment and install dependencies:

```bash
cd AltLoRA

# Create a conda environment
conda create -n prunedlora python=3.9 -y
conda activate prunedlora

# Install dependencies
pip install -r requirements.txt

# Install PEFT in editable mode
pip install -e peft

# Log in to Hugging Face for model and dataset access:
huggingface-cli login
```


### `optim_notes` Argument

The `--optim_notes` flag controls **which optimization or fine-tuning algorithm** is used during training.  
It determines whether the model runs standard full fine-tuning or one of the LoRA-based efficient variants.

#### Available Options

| Value | Algorithm | Description | Recommended Use |
|:------|:-----------|:-------------|:----------------|
| **`adamw`** | *Full Fine-Tuning (AdamW)* | Uses the standard AdamW optimizer to update **all model parameters** (no LoRA modules). Highest memory cost, serves as a baseline. | For small models or full fine-tuning benchmarks. |
| **`lora_rite`** | *LoRA-Rite (Rescaled / Reinitialized LoRA)* | A stable LoRA variant that applies adaptive re-scaling and re-initialization of A/B matrices to improve convergence and training robustness. | Use as a strong, stable LoRA baseline. |
| **`altlora`** | *Alternating LoRA (AltLoRA)* | Alternates updates between the low-rank A/B matrices and frozen backbone parameters. Includes Riemannian projection to preserve low-rank structure. | Default choice — balanced performance and efficiency. |
| **`altlora_plus`** | *AltLoRA+ (Dynamic Rank & Adaptive Scaling)* | Extends AltLoRA with **dynamic rank allocation** and **adaptive scaling** (μ-projection) for better generalization and faster convergence. | Recommended for large models, instruction tuning, or multi-round/federated training. |

#### Usage Example

In `run_train.sh`, set the optimizer strategy:
```bash
optim_notes=altlora
