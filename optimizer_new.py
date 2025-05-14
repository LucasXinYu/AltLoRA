import torch
import torch.optim as optim
import copy
import math
from torch.optim import Optimizer

class SGDr(optim.Optimizer):
    def __init__(self, model, lr, weight_decay=0.0, rank=4, reg=1e-6):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.model = model
        self.rank = rank
        self.reg = reg
        self.learning_rate = lr
        self.weight_decay = weight_decay
        params = list(model.parameters())
        defaults = dict(lr=lr, weight_decay=weight_decay, rank=rank, reg=reg)
        super(SGDr, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        named_params = list(self.model.named_parameters())
        i = 0
        while i < len(named_params):
            name1, p1 = named_params[i]
            if name1.endswith("lora_A.default.weight"):  
                        name2, p2 = named_params[i + 1]
                        i+=1
                        if p1.grad is None and p2.grad is None:
                            continue
                        dim_1 = p2.data.shape[0] // 2

                        if p1.grad is not None:
                            # update p1
                            grad1_0 = p1.grad.data[0:self.rank, :]
                            grad1_1 = p1.grad.data[self.rank:, :]
                            scale1_0 = p2.data[0:dim_1, :]
                            scale1_1 = p2.data[dim_1:, :]

                            try:
                                grad1_0_scaled = torch.inverse(scale1_0.T @ scale1_0 + self.reg * torch.eye(self.rank).to(scale1_0.device)) @ grad1_0
                            except:
                                grad1_0_scaled = grad1_0

                            try:
                                grad1_1_scaled = torch.inverse(scale1_1.T @ scale1_1 + self.reg * torch.eye(self.rank).to(scale1_1.device)) @ grad1_1
                            except:
                                grad1_1_scaled = grad1_1

                            grad1_scaled = torch.cat([grad1_0_scaled, grad1_1_scaled])
                            p1.data.add_(grad1_scaled, alpha=-self.learning_rate)
                            if self.weight_decay > 0.0:
                                p1.data.add_(p1.data, alpha=-self.learning_rate * self.weight_decay)

                        if p2.grad is not None:
                            # update p2
                            grad2_0 = p2.grad.data[0:dim_1, :]
                            grad2_1 = p2.grad.data[dim_1:, :]
                            scale2_0 = p1.data[0:self.rank, :]
                            scale2_1 = p1.data[self.rank:, :]

                            try:
                                grad2_0_scaled = grad2_0 @ torch.inverse(scale2_0 @ scale2_0.T + self.reg * torch.eye(self.rank).to(scale2_0.device))
                            except:
                                grad2_0_scaled = grad2_0

                            try:
                                grad2_1_scaled = grad2_1 @ torch.inverse(scale2_1 @ scale2_1.T + self.reg * torch.eye(self.rank).to(scale2_1.device))
                            except:
                                grad2_1_scaled = grad2_1

                            grad2_scaled = torch.cat([grad2_0_scaled, grad2_1_scaled])
                            p2.data.add_(grad2_scaled, alpha=-self.learning_rate)
                            if self.weight_decay > 0.0:
                                p2.data.add_(p2.data, alpha=-self.learning_rate * self.weight_decay)
            i += 1

        return loss




class altlora(Optimizer):
    def __init__(self, model, lr=1e-3, betas=(0.999, 0.98), eps=1e-6, weight_decay=1e-5, correct_bias=True, rank=2, reg=1e-3):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        params = list(model.parameters())
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super(altlora, self).__init__(params, defaults)
        self.model = model
        self.rank = rank
        self.reg = reg
    def reset_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        named_params = list(self.model.named_parameters())
        i = 0
        while i < len(named_params):
            name1, p1 = named_params[i]
            if name1.endswith("lora_A.default.weight"): 
                if i + 1 < len(named_params):
                    name2, p2 = named_params[i + 1]
                    i += 1  
                    if p1.grad is not None or p2.grad is not None:
                        k = p2.data.shape[0]
                        d = p1.data.shape[1]
                        r = p2.data.shape[1]
                        if p1.grad is not None:
                            state = self.state[p1]
                            if len(state) == 0:
                                state["step"] = 0
                                state["exp_avg"] = torch.zeros(r, d)   
                                state['p2old'] = torch.zeros(k, r)
                            exp_avg = state["exp_avg"].to(p1.device)
                            p2_old = state['p2old'].to(p1.device)
                            beta1, beta2 = self.defaults["betas"]
                            state["step"] += 1                             
                            try:
                                b_precon = torch.inverse(p2.data.T @ p2.data + self.reg * torch.eye(self.rank).to(p2.data.device))
                                P_at = b_precon  @ p2.data.T @ p2_old 
                            except:
                                b_precon = torch.eye((p2.data.T @ p2.data).shape[0]).to(p2.data.device)
                                P_at = b_precon  @ p2.data.T @ p2_old 
                                print('no inversep2')

                            grad1_scaled = b_precon @ p1.grad.data
                            assert grad1_scaled.shape == p1.grad.data.shape

                            exp_avg = beta1 * P_at @ exp_avg + (1-beta1) * grad1_scaled
                            step_size = self.defaults["lr"]
                            if self.defaults["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                step_size = step_size / bias_correction1
                            
                            p1.data = p1.data - step_size * exp_avg 

                            if self.defaults["weight_decay"] > 0.0:
                                p1.data.add_(p1.data, alpha=-self.defaults["lr"] * self.defaults["weight_decay"])
                            state["exp_avg"] = exp_avg.detach().clone()
                            state['p2old'] = p2.data.detach().clone()
                            
                            # Eigenvalues of p1 p1^T
                            p1_cov = p1.data @ p1.data.T
                            eigvals_p1 = torch.linalg.eigvalsh(p1_cov).cpu()
                            print(f"[Eig] Eigenvalues of p1 p1^T at step {state['step']}: {eigvals_p1}")

                        if p2.grad is not None:
                            state = self.state[p2]
                            if len(state) == 0:
                                state["step"] = 0
                                state["exp_avg"] = torch.zeros(k, r)
                                state['p1old'] = torch.zeros(r, d)

                            exp_avg= state["exp_avg"].to(p2.device)
                            p1_old = state['p1old'].to(p2.device)
                            beta1, beta2 = self.defaults["betas"]
                            state["step"] += 1
                            
                           
                            try:
                                a_precon = torch.inverse(p1.data @ p1.data.T + self.reg * torch.eye(self.rank).to(p1.data.device))
                                P_bt = p1_old @ p1.data.T @ a_precon
                            except:
                                a_precon = torch.eye((p1.data @ p1.data.T).shape[0]).to(p1.data.device) 
                                P_bt = p1_old @ p1.data.T @ a_precon
                                print('no inverse')
                             
                 
                            grad2_scaled = p2.grad.data @ a_precon 
                            assert grad2_scaled.shape == p2.grad.data.shape
                            exp_avg = beta1 * exp_avg @ P_bt + (1-beta1) * grad2_scaled
                            step_size = self.defaults["lr"]
                            if self.defaults["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                step_size = step_size / bias_correction1
                            
                            p2.data = p2.data - step_size * exp_avg 

                            if self.defaults["weight_decay"] > 0.0:
                                p2.data.add_(p2.data, alpha=-self.defaults["lr"] * self.defaults["weight_decay"])
                            state["exp_avg"] = exp_avg.detach().clone()
                            state['p1old'] = p1.data.detach().clone()
                            
                            p2_cov = -p2.data.T @ p2.data
                            eigvals_p2 = torch.linalg.eigvalsh(p2_cov).cpu()
                            print(f"[Eig] Eigenvalues of -p2^T p2 at step {state['step']}: {eigvals_p2}")

            i += 1

        return loss

class altlora_plus(Optimizer):
    def __init__(self, model, lr=1e-3, betas=(0.999, 0.98), eps=1e-6, weight_decay=1e-5, correct_bias=True, rank=2, reg=1e-3):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        params = list(model.parameters())
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super(altlora_plus, self).__init__(params, defaults)
        self.model = model
        self.rank = rank
        self.reg = reg
    def reset_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        named_params = list(self.model.named_parameters())
        i = 0
        while i < len(named_params):
            name1, p1 = named_params[i]
            if name1.endswith("lora_A.default.weight"):  
                if i + 1 < len(named_params):
                    name2, p2 = named_params[i + 1]
                    i += 1 
                    if p1.grad is not None or p2.grad is not None:
                        k = p2.data.shape[0]
                        d = p1.data.shape[1]
                        r = p2.data.shape[1]
                        if p1.grad is not None:
                            state = self.state[p1]
                            if len(state) == 0:
                                state["step"] = 0
                                state["exp_avg"] = torch.zeros(r, d) 
                                state["exp_avg_sq"] = torch.zeros(r, d)  
                                state['p2old'] = torch.zeros(k, r)
                            exp_avg = state["exp_avg"].to(p1.device)
                            p2_old = state['p2old'].to(p1.device)
                            exp_avg_sq = state["exp_avg_sq"].to(p1.device)
                            beta1, beta2 = self.defaults["betas"]
                            state["step"] += 1
                            try:
                                b_precon = torch.inverse(p2.data.T @ p2.data + self.reg * torch.eye(self.rank).to(p2.data.device))
                                P_at = b_precon  @ p2.data.T @ p2_old 
                            except:
                                b_precon = torch.eye((p2.data.T @ p2.data).shape[0]).to(p2.data.device)
                                P_at = b_precon  @ p2.data.T @ p2_old 
                                print('no inversep2')

                            grad1_scaled = b_precon @ p1.grad.data
                            assert grad1_scaled.shape == p1.grad.data.shape
                            
                            step_size = self.defaults["lr"]
                            if self.defaults["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                step_size = step_size / bias_correction1
                            exp_avg_sq.mul_(beta2).addcmul_(grad1_scaled, grad1_scaled, value=1.0 - beta2)
                            denom = exp_avg_sq.sqrt().add_(self.defaults["eps"])

                            p1.data.addcdiv_(-step_size, beta1 * P_at @ exp_avg + (1-beta1) * grad1_scaled,denom)

                            if self.defaults["weight_decay"] > 0.0:
                                p1.data.add_(p1.data, alpha=-self.defaults["lr"] * self.defaults["weight_decay"])
                            exp_avg = beta1 * exp_avg + (1-beta1) * grad1_scaled
                            state["exp_avg"] = exp_avg.detach().clone()
                            state['p2old'] = p2.data.detach().clone()
                            state['exp_avg_sq'] =  exp_avg_sq.detach().clone()
                        if p2.grad is not None:
                            state = self.state[p2]
                            if len(state) == 0:
                                state["step"] = 0
                                state["exp_avg"] = torch.zeros(k, r)
                                state["exp_avg_sq"] = torch.zeros(k, r)  
                                state['p1old'] = torch.zeros(r, d)
                            exp_avg= state["exp_avg"].to(p2.device)
                            exp_avg_sq = state['exp_avg_sq'].to(p2.device)
                            p1_old = state['p1old'].to(p2.device)
                            beta1, beta2 = self.defaults["betas"]
                            state["step"] += 1
                            try:
                                a_precon = torch.inverse(p1.data @ p1.data.T + self.reg * torch.eye(self.rank).to(p1.data.device))
                                P_bt = p1_old @ p1.data.T @ a_precon
                            except:
                                a_precon = torch.eye((p1.data @ p1.data.T).shape[0]).to(p1.data.device) 
                                P_bt = p1_old @ p1.data.T @ a_precon
                                print('no inverse')
                            grad2_scaled = p2.grad.data @ a_precon 
                            assert grad2_scaled.shape == p2.grad.data.shape
                            step_size = self.defaults["lr"]
                            if self.defaults["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                step_size = step_size / bias_correction1
                            exp_avg_sq.mul_(beta2).addcmul_(grad2_scaled, grad2_scaled, value=1.0 - beta2)
                            denom = exp_avg_sq.sqrt().add_(self.defaults["eps"])
                            p2.data.addcdiv_(-step_size, beta1 * exp_avg @ P_bt + (1-beta1) * grad2_scaled,denom)
                            if self.defaults["weight_decay"] > 0.0:
                                p2.data.add_(p2.data, alpha=-self.defaults["lr"] * self.defaults["weight_decay"])
                            exp_avg = beta1 * exp_avg  + (1-beta1) * grad2_scaled
                            state["exp_avg"] = exp_avg.detach().clone()
                            state['exp_avg_sq'] =  exp_avg_sq.detach().clone()
                            state['p1old'] = p1.data.detach().clone()
            i += 1
        return loss

 
