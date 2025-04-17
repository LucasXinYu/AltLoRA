from typing import Literal,Callable, Dict, List, Optional, Tuple, Union, Any
import torch
import wandb
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import get_scheduler
from transformers import Trainer,Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.trainer import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
import torch
import torch.optim as optim
from peft.tuners.lora.layer import Linear as LoraLinear
from optimizer_new import *
# include_keywords = ["block.0", "block.4"]
include_keywords = ["encoder.block.2", "encoder.block.3", "encoder.block.4"]  # for T5
# include_keywords = ["layers.27", "layers.6"]  # for Llama
do_log = False


def get_forward_hook(name):
    def hook(module, input, output):
        wandb.log(
            {
                f"{name}/input_mean": input[0].mean().item(),
                f"{name}/input_std": input[0].std().item(),
                f"{name}/output_mean": output.mean().item(),
                f"{name}/output_std": output.std().item(),
            },
            commit=False,
        )

    return hook

class LogTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Seq2SeqTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        optim_name: Optional[Literal["Adamw", "gd", "scaledgd","adamwr"]] = None,  # 可以是 "adamw", "gd", "scaledgd" 或 None
        alt: Optional[bool] = None,  # 只能是 bool 或 None
        rank: Optional[int] = None,
    ):  
      
        super().__init__(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        model_init=model_init,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=optimizers,  
        preprocess_logits_for_metrics=preprocess_logits_for_metrics, 
    )
        self.is_peft = "PeftModel" in type(model).__name__
        print('==========================================')
        print(self.is_peft)
        print('==========================================')
        if self.is_peft:
            for name, module in model.named_modules():
                if isinstance(module, LoraLinear):
                    self.scaling = module.scaling["default"]
                    break
        self.orig_A = None
        self.orig_B = None
        self.orig_W = None
        self.gradient_accumulation_counter = 0
        self.update_A = True  
        self.lr = args.learning_rate
        self.rank = rank
        self.optim_name = optim_name
        self.alt = alt

    
        print('=================================================================================')
        print(optim_name)
        print(self.rank)
        print(self.alt)
        print('=================================================================================')
        print('=================================================================================')
        print(self.args.lr_scheduler_type)
        print('=================================================================================')
        if optimizers == (None, None):
            if self.optim_name is not None and self.optim_name != "Adamw":
                print('create_optimizer')
                optimizers = self.create_optimizer()
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
                scheduler = get_scheduler(
                    name=args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=args.get_warmup_steps(args.max_steps),
                    num_training_steps=args.max_steps,
                )
                optimizers = (optimizer, scheduler)


    def freeze_A(self, model):
        for name, param in model.named_parameters():
            if "lora_A" in name:
                param.requires_grad = False
                param.grad = None

    def freeze_B(self, model):
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.requires_grad = False
                param.grad = None

    def unfreeze_A(self, model):
        for name, param in model.named_parameters():
            if "lora_A" in name:
                param.requires_grad = True

    def unfreeze_B(self, model):
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.requires_grad = True


    def create_optimizer(self):
        if self.train_dataset is None:
            raise ValueError("train_dataset must be provided to compute training steps for scheduler")

        # Compute total steps
        if self.args.max_steps > 0:
            num_training_steps = self.args.max_steps
        else:
            train_batch_size = self.args.train_batch_size
            gradient_accumulation = self.args.gradient_accumulation_steps
            epoch_steps = len(self.train_dataset) // train_batch_size
            num_training_steps = epoch_steps * self.args.num_train_epochs // gradient_accumulation

        # Get warmup steps
        num_warmup_steps = self.args.get_warmup_steps(num_training_steps)

        # Define optimizer
        if self.optim_name == "gd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=1e-2
            )

        elif self.optim_name == "scaledgd":
            optimizer = SGDr(
                self.model,
                lr=self.lr,
                weight_decay=1e-3,
                rank=self.rank,
                reg=1e-6
            )

        elif self.optim_name == "adamwr":
            optimizer = AdamWr_fm1(
                self.model,
                lr=self.lr,
                betas=(0.9, 0.999),
                eps=1e-4,
                weight_decay=1e-5,
                correct_bias=True,
                rank=self.rank,
                reg=1e-5
            )
        elif self.optim_name == "Adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        # Hugging Face scheduler
        scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,  # e.g., "cosine", "linear"
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        return optimizer, scheduler

  


    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        #print(self.gradient_accumulation_counter)
        if not do_log:
            #print(self.args.gradient_accumulation_steps)
        
            if self.alt is True:
                #print(self.args.gradient_accumulation_steps)  128
                self.gradient_accumulation_counter += 1
                if ( self.gradient_accumulation_counter % (10 *self.args.gradient_accumulation_steps) == 0 ):
                    if self.update_A :
                        print("update_A")
                        self.freeze_B(model)  # 冻结 B
                        self.unfreeze_A(model)  # 解冻 A
                    else:        
                        print("update_B")
                        self.freeze_A(model)  # 冻结 A
                        self.unfreeze_B(model)  # 解冻 B
                    self.update_A = not self.update_A
                
            return super().training_step(model, inputs)


        
        #self.update_A = not self.update_A

        if self.is_peft:
            if self.orig_A is None:
                self.orig_A = {}
                self.orig_B = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and any(
                        [kw in name for kw in include_keywords]
                    ):
                        if "lora_A" in name:
                            self.orig_A[name.split("lora_A.")[0]] = (
                                param.detach().clone()
                            )
                        elif "lora_B" in name:
                            self.orig_B[name.split("lora_B.")[0]] = (
                                param.detach().clone()
                            )
                for name, module in model.named_modules():
                    if any([kw in name for kw in include_keywords]) and isinstance(
                        module, LoraLinear
                    ):
                        breakpoint()
                        hook = get_forward_hook(name)
                        module.register_forward_hook(hook)
        else:
            if self.orig_W is None:
                self.orig_W = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and any(
                        [kw in name for kw in include_keywords]
                    ):
                        self.orig_W[name] = param.detach().clone()

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)

        with torch.no_grad():
            if (
                self.gradient_accumulation_counter
                % self.args.gradient_accumulation_steps
                == self.args.gradient_accumulation_steps - 1
            ):
                if self.is_peft:
                    A_dict = {}
                    B_dict = {}
                    for name, param in model.named_parameters():
                        if param.requires_grad and any(
                            [kw in name for kw in include_keywords]
                        ):
                            if "lora_A" in name:
                                A_dict[name.split("lora_A.")[0]] = param
                            elif "lora_B" in name:
                                B_dict[name.split("lora_B.")[0]] = param
                    assert (
                        len(A_dict)
                        == len(self.orig_A)
                        == len(B_dict)
                        == len(self.orig_B)
                    ), (
                        len(A_dict),
                        len(self.orig_A),
                        len(B_dict),
                        len(self.orig_B),
                    )
                    for key in A_dict.keys():
                        A = A_dict[key]
                        B = B_dict[key]
                        lora_r = A.shape[0]
                        A_grad = A_dict[key].grad
                        B_grad = B_dict[key].grad
                        A_0 = self.orig_A[key]
                        B_0 = self.orig_B[key]
                        A_diff = A - A_0
                        B_diff = B - B_0
                        BA = torch.matmul(B, A)
                        BA_0 = torch.matmul(B_0, A_0)
                        BA_diff = BA - BA_0
                        BA_diff_norm = torch.norm(BA_diff).item()
                        A_diff_norm = torch.norm(A_diff).item()
                        B_diff_norm = torch.norm(B_diff).item()
                        A_norm = torch.norm(A).item()
                        B_norm = torch.norm(B).item()
                        A_grad_norm = torch.norm(A_grad).item()
                        B_grad_norm = torch.norm(B_grad).item()
                        BA_singular_values = torch.svd_lowrank(
                            BA_diff.float(), q=2 * lora_r
                        )[1][:lora_r]
                        top_1_ratio = (
                            BA_singular_values[0] / BA_singular_values.sum()
                        ).item()
                        top_4_ratio = (
                            BA_singular_values[:4].sum() / BA_singular_values.sum()
                        ).item()
                        wandb.log(
                            {
                                f"A_norm/{key}": A_norm,
                                f"B_norm/{key}": B_norm,
                                f"A_grad_norm/{key}": A_grad_norm,
                                f"B_grad_norm/{key}": B_grad_norm,
                                f"A_diff_norm/{key}": A_diff_norm,
                                f"B_diff_norm/{key}": B_diff_norm,
                                f"BA_diff_norm/{key}": BA_diff_norm,
                                f"scaled_BA_diff_norm/{key}": self.scaling
                                * BA_diff_norm,
                                f"BA_top_1_ratio/{key}": top_1_ratio,
                                f"BA_top_4_ratio/{key}": top_4_ratio,
                                "train/global_step": self.state.global_step,
                            }
                        )
                else:
                    W_dict = {}
                    for name, param in model.named_parameters():
                        if (
                            param.requires_grad
                            and any([kw in name for kw in include_keywords])
                            and len(param.shape) == 2
                        ):
                            W_dict[name] = param
                    for key in W_dict.keys():
                        W = W_dict[key]
                        W_grad = W.grad
                        W_0 = self.orig_W[key]
                        W_diff = W - W_0
                        W_diff_norm = torch.norm(W_diff).item()
                        W_norm = torch.norm(W).item()
                        W_grad_norm = torch.norm(W_grad).item()
                        U, S, V = torch.svd(W_diff.float())
                        top_1_ratio = S[0] / S.sum()
                        top_4_ratio = S[:4].sum() / S.sum()
                        wandb.log(
                            {
                                f"W_norm/{key}": W_norm,
                                f"W_grad_norm/{key}": W_grad_norm,
                                f"W_diff_norm/{key}": W_diff_norm,
                                "train/global_step": self.state.global_step,
                                f"W_top_1_ratio/{key}": top_1_ratio.item(),
                                f"W_top_4_ratio/{key}": top_4_ratio.item(),
                            }
                        )
        self.gradient_accumulation_counter += 1

        return loss.detach() / self.args.gradient_accumulation_steps