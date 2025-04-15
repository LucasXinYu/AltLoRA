import torch
import torch.optim as optim
import copy

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

        # Collect all parameters from the model
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
            if name1.endswith("lora_A.default.weight"):  # check lora_A
                #if i + 1 < len(named_params):
                        name2, p2 = named_params[i + 1]
                    #if name2.endswith("lora_B.default.weight") and name2.replace("lora_B.default.weight", "lora_A.default.weight") == name1:
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

                            # Apply weight decay to p1
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

                            # Apply weight decay to p2
                            if self.weight_decay > 0.0:
                                p2.data.add_(p2.data, alpha=-self.learning_rate * self.weight_decay)
            i += 1

        return loss



import torch
import math
from torch.optim import Optimizer

class AdamWr(Optimizer):
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0, correct_bias=True, rank=2, reg=1e-6):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))

        # Collect all parameters from the model
        params = list(model.parameters())
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super(AdamWr, self).__init__(params, defaults)

        self.model = model
        self.rank = rank
        self.reg = reg
        print(f'{self.reg=}')

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
            if name1.endswith("lora_A.default.weight"):  # Check if the parameter name matches
                if i + 1 < len(named_params):
                    name2, p2 = named_params[i + 1]
                    i += 1  # Move to the next pair

                    # Only proceed if gradients are not None
                    if p1.grad is not None or p2.grad is not None:
                        print(p1.grad == None)
                        print(p2.grad == None)
                        #print(name1,name2)
                        # Update p1
                        if p1.grad is not None:
                            state = self.state[p1]
                            if len(state) == 0:
                                state["step"] = 0
                                state["exp_avg"] = torch.zeros_like(p1.data)
                                state["exp_avg_sq"] = torch.zeros_like(p1.data)

                            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                            beta1, beta2 = self.defaults["betas"]
                            state["step"] += 1

                            dim_1 = p2.data.shape[0] // 2
                            ###
                            grad1_scaled0 = p1.grad.data[0:self.rank, :]
                            c0 = p2.data[0:dim_1, :]
                            try:
                                grad1_scaled0 = torch.inverse(c0.T @ c0 + self.reg * torch.eye(self.rank).to(c0.device))@ grad1_scaled0
                            except:
                                grad1_scaled0 = grad1_scaled0
                            
                            ###
                            grad1_scaled1 = p1.grad.data[self.rank:, :]
                            c1 = p2.data[dim_1:, :]
                            try:
                                grad1_scaled1 = torch.inverse(c1.T @ c1 + self.reg * torch.eye(self.rank).to(c1.device)) @ grad1_scaled1
                            except:
                                grad1_scaled1 = grad1_scaled1

                            grad1_scaled = torch.cat([grad1_scaled0, grad1_scaled1])
                            assert grad1_scaled.shape == p1.grad.data.shape

                            exp_avg.mul_(beta1).add_(grad1_scaled, alpha=1.0 - beta1)
                            exp_avg_sq.mul_(beta2).addcmul_(grad1_scaled, grad1_scaled, value=1.0 - beta2)
                            denom = exp_avg_sq.sqrt().add_(self.defaults["eps"])

                            step_size = self.defaults["lr"]
                            if self.defaults["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                bias_correction2 = 1.0 - beta2 ** state["step"]
                                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                            p1.data.addcdiv_(-step_size, exp_avg, denom)
                            if self.defaults["weight_decay"] > 0.0:
                                p1.data.add_(p1.data, alpha=-self.defaults["lr"] * self.defaults["weight_decay"])

                        # Update p2
                        if p2.grad is not None:
                            state = self.state[p2]
                            if len(state) == 0:
                                state["step"] = 0
                                state["exp_avg"] = torch.zeros_like(p2.data)
                                state["exp_avg_sq"] = torch.zeros_like(p2.data)

                            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                            beta1, beta2 = self.defaults["betas"]
                            state["step"] += 1
                            
                            ##
                            grad2_scaled0 = p2.grad.data[0:dim_1, :]
                            c0 = p1.data[0:self.rank, :]
                            try:
                                grad2_scaled0 = grad2_scaled0 @ torch.inverse(c0 @ c0.T + self.reg * torch.eye(self.rank).to(c0.device))
                            except:
                                grad2_scaled0 = grad2_scaled0 
                             
                            ##
                            grad2_scaled1 = p2.grad.data[dim_1:, :]
                            c1 = p1.data[self.rank:, :]
                            try:
                                grad2_scaled1 = grad2_scaled1 @torch.inverse(c1 @ c1.T + self.reg * torch.eye(self.rank).to(c1.device))
                            except:
                                grad2_scaled1 = grad2_scaled1
                            grad2_scaled = torch.cat([grad2_scaled0, grad2_scaled1])
                            assert grad2_scaled.shape == p2.grad.data.shape

                            exp_avg.mul_(beta1).add_(grad2_scaled, alpha=1.0 - beta1)
                            exp_avg_sq.mul_(beta2).addcmul_(grad2_scaled, grad2_scaled, value=1.0 - beta2)
                            denom = exp_avg_sq.sqrt().add_(self.defaults["eps"])

                            step_size = self.defaults["lr"]
                            if self.defaults["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                bias_correction2 = 1.0 - beta2 ** state["step"]
                                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                            p2.data.addcdiv_(-step_size, exp_avg, denom)
                            if self.defaults["weight_decay"] > 0.0:
                                p2.data.add_(p2.data, alpha=-self.defaults["lr"] * self.defaults["weight_decay"])
            i += 1

        return loss

#####################################################################################


class AdamWr_full(Optimizer):
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=1e-5, correct_bias=True, rank=2, reg=1e-6):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))

        # Collect all parameters from the model
        params = list(model.parameters())
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super(AdamWr_full, self).__init__(params, defaults)

        self.model = model
        self.rank = rank
        self.reg = reg
        print(f'{self.reg=}')

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
            if name1.endswith("lora_A.default.weight"):  # Check if the parameter name matches
                if i + 1 < len(named_params):
                    name2, p2 = named_params[i + 1]
                    i += 1  # Move to the next pair

                    # Only proceed if gradients are not None
                    if p1.grad is not None or p2.grad is not None:
                        k = p2.grad.shape[0]
                        d = p1.grad.shape[1]
                        # Update p1
                        if p1.grad is not None:
                            state = self.state[p1]
                            if len(state) == 0:
                                state["step"] = 0
                                state["exp_avg"] = torch.empty(k, d)
                                state["exp_avg_sq"] = torch.empty(k, d)

                            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                            beta1, beta2 = self.defaults["betas"]
                            state["step"] += 1

                            dim_1 = p2.data.shape[0] // 2
                            ###
                            grad1_scaled0 = p1.grad.data[0:self.rank, :]
                            c0 = p2.data[0:dim_1, :]
                            try:
                                grad1_scaled0 = torch.inverse(c0.T @ c0 + self.reg * torch.eye(self.rank).to(c0.device))@ grad1_scaled0
                            except:
                                grad1_scaled0 = grad1_scaled0
                            
                            ###
                            grad1_scaled1 = p1.grad.data[self.rank:, :]
                            c1 = p2.data[dim_1:, :]
                            try:
                                grad1_scaled1 = torch.inverse(c1.T @ c1 + self.reg * torch.eye(self.rank).to(c1.device)) @ grad1_scaled1
                            except:
                                grad1_scaled1 = grad1_scaled1

                            grad1_scaled = torch.cat([grad1_scaled0, grad1_scaled1])  #r d
                            assert grad1_scaled.shape == p1.grad.data.shape

                            full_grad1_scaled = (p2.data @ grad1_scaled).to(c1.device)
                            exp_avg = exp_avg.to(c1.device)
                            exp_avg_sq = exp_avg_sq.to(c1.device)

                            exp_avg.mul_(beta1).add_(full_grad1_scaled, alpha=1.0 - beta1)  #k d
                            exp_avg_sq.mul_(beta2).addcmul_(full_grad1_scaled, full_grad1_scaled, value=1.0 - beta2)  #k d
                            denom = exp_avg_sq.sqrt().add_(self.defaults["eps"])

                            step_size = self.defaults["lr"]
                            if self.defaults["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                bias_correction2 = 1.0 - beta2 ** state["step"]
                                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                            
                            preconditioner =  torch.inverse(p2.data.T @ p2.data + self.reg * torch.eye(self.rank).to(c0.device)) @ p2.data.T  #r k
                            p1.data = p1.data - step_size * preconditioner @ torch.div(exp_avg, denom)

                            #p1.data.addcdiv_(, preconditioner @ exp_avg, denom)
                            if self.defaults["weight_decay"] > 0.0:
                                p1.data.add_(p1.data, alpha=-self.defaults["lr"] * self.defaults["weight_decay"])

                        # Update p2
                        if p2.grad is not None:
                            state = self.state[p2]
                            if len(state) == 0:
                                state["step"] = 0
                                state["exp_avg"] = torch.empty(k, d)
                                state["exp_avg_sq"] = torch.empty(k, d)

                            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                            beta1, beta2 = self.defaults["betas"]
                            state["step"] += 1
                            
                            ##
                            grad2_scaled0 = p2.grad.data[0:dim_1, :]
                            c0 = p1.data[0:self.rank, :]
                            try:
                                grad2_scaled0 = grad2_scaled0 @ torch.inverse(c0 @ c0.T + self.reg * torch.eye(self.rank).to(c0.device))
                            except:
                                grad2_scaled0 = grad2_scaled0 
                             
                            ##
                            grad2_scaled1 = p2.grad.data[dim_1:, :]
                            c1 = p1.data[self.rank:, :]
                            try:
                                grad2_scaled1 = grad2_scaled1 @torch.inverse(c1 @ c1.T + self.reg * torch.eye(self.rank).to(c1.device))
                            except:
                                grad2_scaled1 = grad2_scaled1
                            grad2_scaled = torch.cat([grad2_scaled0, grad2_scaled1])  # kr
                            assert grad2_scaled.shape == p2.grad.data.shape

                            full_grad1_scaled = (grad2_scaled @ p1.data).to(c1.device) # kr rd
                            exp_avg = exp_avg.to(c1.device)
                            exp_avg_sq = exp_avg_sq.to(c1.device)

                            exp_avg.mul_(beta1).add_(full_grad1_scaled, alpha=1.0 - beta1)
                            exp_avg_sq.mul_(beta2).addcmul_(full_grad1_scaled, full_grad1_scaled, value=1.0 - beta2)
                            denom = exp_avg_sq.sqrt().add_(self.defaults["eps"])

                            step_size = self.defaults["lr"]
                            if self.defaults["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                bias_correction2 = 1.0 - beta2 ** state["step"]
                                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                            
                            preconditioner =  p1.data.T @ torch.inverse(p1.data @ p1.data.T + self.reg * torch.eye(self.rank).to(c0.device)) #d r 
                            p2.data = p2.data - step_size * torch.div(exp_avg, denom) @ preconditioner  # kd dr 

                            #p2.data.addcdiv_(-step_size, exp_avg, denom)
                            if self.defaults["weight_decay"] > 0.0:
                                p2.data.add_(p2.data, alpha=-self.defaults["lr"] * self.defaults["weight_decay"])
            i += 1

        return loss



#####################################################################################



class AdamWr_fm(Optimizer):
    def __init__(self, model, lr=1e-3, betas=(0.999, 0.98), eps=1e-6, weight_decay=1e-5, correct_bias=True, rank=2, reg=1e-3):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))

        # Collect all parameters from the model
        params = list(model.parameters())
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super(AdamWr_fm, self).__init__(params, defaults)

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
            if name1.endswith("lora_A.default.weight"):  # Check if the parameter name matches
                if i + 1 < len(named_params):
                    name2, p2 = named_params[i + 1]
                    i += 1  # Move to the next pair

                    # Only proceed if gradients are not None
                    if p1.grad is not None or p2.grad is not None:
                        k = p2.data.shape[0]
                        d = p1.data.shape[1]
                        r = p2.data.shape[1]
                        # Update p1
                        if p1.grad is not None:
                            state = self.state[p1]
                            if len(state) == 0:
                                state["step"] = 0
                                state["exp_avg"] = torch.zeros(r, d)   
                                #state["exp_avg_sq"] = torch.empty(k, r)   
                                state['p2old'] = torch.zeros(k, r)

                            exp_avg = state["exp_avg"].to(p1.device)
                            p2_old = state['p2old'].to(p1.device)
                            beta1, beta2 = self.defaults["betas"]
                            state["step"] += 1

                            #dim_1 = p2.data.shape[0] // 2
                            ###
                             
                            try:
                                b_precon = torch.inverse(p2.data.T @ p2.data + self.reg * torch.eye(self.rank).to(p2.data.device))
                                P_at = b_precon  @ p2.data.T @ p2_old 
                            except:
                                b_precon = torch.eye((p2.data.T @ p2.data).shape[0]).to(p2.data.device)
                                P_at = b_precon  @ p2.data.T @ p2_old 
                                # print('no inversep2')

                            grad1_scaled = b_precon @ p1.grad.data
                            assert grad1_scaled.shape == p1.grad.data.shape

                            exp_avg = beta1 * P_at @ exp_avg + (1-beta1) * grad1_scaled
                            #exp_avg = beta1 * exp_avg + (1-beta1) * grad1_scaled
                            step_size = self.defaults["lr"]
                            if self.defaults["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                step_size = step_size / bias_correction1
                            
                            p1.data = p1.data - step_size * exp_avg # rk kr rd ->rd

                            if self.defaults["weight_decay"] > 0.0:
                                p1.data.add_(p1.data, alpha=-self.defaults["lr"] * self.defaults["weight_decay"])
                            state["exp_avg"] = exp_avg.detach().clone()
                            state['p2old'] = p2.data.detach().clone()


                        # Update p2
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
                            #exp_avg = beta1 * exp_avg + (1-beta1) * grad2_scaled

                            step_size = self.defaults["lr"]
                            if self.defaults["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                step_size = step_size / bias_correction1
                            
                            p2.data = p2.data - step_size * exp_avg  # kr rd dr  # O(r^2d + r^2k)

                            if self.defaults["weight_decay"] > 0.0:
                                p2.data.add_(p2.data, alpha=-self.defaults["lr"] * self.defaults["weight_decay"])
                            state["exp_avg"] = exp_avg.detach().clone()
                            state['p1old'] = p1.data.detach().clone()

            i += 1

        return loss

#####################################################################################


class adamw_scale(Optimizer):
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0, correct_bias=True, rank=2, reg=1e-6):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))

        # Collect all parameters from the model
        params = list(model.parameters())
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super(adamw_scale, self).__init__(params, defaults)

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
            if name1.endswith("lora_A.default.weight"):  # Check if the parameter name matches
                if i + 1 < len(named_params):
                    name2, p2 = named_params[i + 1]
                    i += 1  # Move to the next pair

                    # Only proceed if gradients are not None
                    if p1.grad is not None or p2.grad is not None:
                        # Update p1
                        if p1.grad is not None:
                            state = self.state[p1]
                            if len(state) == 0:
                                state["step"] = 0
                                state["exp_avg"] = torch.zeros_like(p1.data)
                                state["exp_avg_sq"] = torch.eye(self.rank).to(p1.device)
                                state['p2_previous'] = torch.zeros_like(p2.data)

                            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                            beta1, beta2 = self.defaults["betas"]
                            p2_old = state['p2_previous']
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



                            exp_avg_sq = beta2 * P_at @ exp_avg_sq.to(p1.device) @ P_at.T + (1-beta2) * grad1_scaled @ grad1_scaled.T

                            U, S, V = torch.linalg.svd(exp_avg_sq)
                            sqrt_inv_S = torch.diag(1.0 / torch.sqrt(S)) + self.reg * torch.eye(self.rank).to(p1.device)
                            exp_avg_sq_root = V @ sqrt_inv_S @ U.T

                            
                            S =  exp_avg_sq_root @  grad1_scaled  # r d

                            exp_avg = beta1 * P_at.to(p1.device) @ exp_avg.to(p1.device) + (1 - beta1) * S.to(p1.device)

                            step_size = self.defaults["lr"]
                            if self.defaults["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                bias_correction2 = 1.0 - beta2 ** state["step"]
                                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                            p1.data = p1.data - step_size * exp_avg

                            # Weight decay
                            if self.defaults["weight_decay"] > 0.0:
                                p1.data.add_(p1.data, alpha=-self.defaults["lr"] * self.defaults["weight_decay"])
                            
                            state['p2_previous'] = p2.data.detach().clone()

                        # Update p2
                        if p2.grad is not None:
                            state = self.state[p2]
                            if len(state) == 0:
                                state["step"] = 0
                                state["exp_avg"] = torch.zeros_like(p2.data)
                                state["exp_avg_sq"] = torch.eye(self.rank).to(p2.device)
                                state['p1_previous'] = torch.zeros_like(p1.data)

                            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                            beta1, beta2 = self.defaults["betas"]
                            p1_old = state['p1_previous']
                            state["step"] += 1

                            
                            try:
                                a_precon = torch.inverse(p1.data @ p1.data.T + self.reg * torch.eye(self.rank).to(p1.data.device))
                                P_bt = p1_old @ p1.data.T @ a_precon
                            except:
                                a_precon = torch.eye((p1.data @ p1.data.T).shape[0]).to(p1.data.device) 
                                P_bt = p1_old @ p1.data.T @ a_precon
                                # print('no inverse')
                             
                 
                            grad2_scaled = p2.grad.data @ a_precon 
                            assert grad2_scaled.shape == p2.grad.data.shape

                            exp_avg_sq = beta2 * P_bt.T @ exp_avg_sq.to(p1.device) @ P_bt + (1-beta2) * grad2_scaled.T @ grad2_scaled

                            U, S, V = torch.linalg.svd(exp_avg_sq)
                            sqrt_inv_S = torch.diag(1.0 / torch.sqrt(S)) + self.reg * torch.eye(self.rank).to(p1.device)
                            exp_avg_sq_root = V @ sqrt_inv_S @ U # r r

                            S =  grad2_scaled @ exp_avg_sq_root 
                            exp_avg =  beta1 * exp_avg.to(p1.device) @ P_bt.to(p1.device) + (1-beta1) * S.to(p1.device)

                            step_size = self.defaults["lr"]
                            if self.defaults["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                bias_correction2 = 1.0 - beta2 ** state["step"]
                                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                            p2.data = p2.data - step_size * exp_avg

                            # Weight decay
                            if self.defaults["weight_decay"] > 0.0:
                                p2.data.add_(p2.data, alpha=-self.defaults["lr"] * self.defaults["weight_decay"])

                            state['p1_previous'] = p1.data.detach().clone()

            i += 1

        return loss

###############
###second moment only multiple with current gradient 
#### hist_first + cumm_scone * current_grad

class adamw_scale_old(Optimizer):
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0, correct_bias=True, rank=2, reg=1e-6):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))

        # Collect all parameters from the model
        params = list(model.parameters())
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super(adamw_scale_old, self).__init__(params, defaults)

        self.model = model
        self.rank = rank
        self.reg = reg
        print(f'{self.reg=}')

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
            if name1.endswith("lora_A.default.weight"):  # Check if the parameter name matches
                if i + 1 < len(named_params):
                    name2, p2 = named_params[i + 1]
                    i += 1  # Move to the next pair

                    # Only proceed if gradients are not None
                    if p1.grad is not None or p2.grad is not None:
                        # Update p1
                        if p1.grad is not None:
                            state = self.state[p1]
                            if len(state) == 0:
                                state["step"] = 0
                                state["exp_avg"] = torch.zeros_like(p1.data)
                                state["exp_avg_sq"] = torch.eye(self.rank).to(p1.device)
                                state['p2_previous'] = torch.ones_like(p2.data)

                            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                            beta1, beta2 = self.defaults["betas"]
                            p2_previous = state['p2_previous']
                            state["step"] += 1

                            dim_1 = p2.data.shape[0] // 2

                            ###
                            grad1_scaled0 = p1.grad.data[0:self.rank, :]
                            c0 = p2.data[0:dim_1, :]
                            try:
                                grad1_scaled0 = torch.inverse(c0.T @ c0 + self.reg * torch.eye(self.rank).to(c0.device)) @ grad1_scaled0
                            except:
                                grad1_scaled0 = grad1_scaled0

                            ###
                            grad1_scaled1 = p1.grad.data[self.rank:, :]
                            c1 = p2.data[dim_1:, :]
                            try:
                                grad1_scaled1 = torch.inverse(c1.T @ c1 + self.reg * torch.eye(self.rank).to(c1.device)) @ grad1_scaled1
                            except:
                                grad1_scaled1 = grad1_scaled1

                            grad1_scaled = torch.cat([grad1_scaled0, grad1_scaled1])
                            assert grad1_scaled.shape == p1.grad.data.shape

                            # r r
                            BB_inverse = torch.inverse(p2.data.T @ p2.data + self.reg * torch.eye(self.rank).to(c0.device))
                            # rr rk kr --> rr
                            P_at = BB_inverse @ p2.data.T @ p2_previous
                            P_at = P_at.to(c0.device)
                            # rr second moment
                            exp_avg_sq = 0.5 * exp_avg_sq.to(c0.device) + 0.5 * grad1_scaled @ grad1_scaled.T
                            ######################
                            #exp_avg_sq = torch.eye(self.rank).to(c0.device)
                            ######################
                            U, S, V = torch.linalg.svd(exp_avg_sq)
                            sqrt_inv_S = torch.diag(1.0 / torch.sqrt(S)) + self.reg * torch.eye(self.rank).to(c0.device)
                            exp_avg_sq_root = V @ sqrt_inv_S @ U.T

                            S =  exp_avg_sq_root @ grad1_scaled  # r d

                            exp_avg = 0.9 * exp_avg.to(c0.device) + 0.1 * S.to(c0.device)

                            step_size = self.defaults["lr"]
                            if self.defaults["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                bias_correction2 = 1.0 - beta2 ** state["step"]
                                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                            p1.data = p1.data - step_size * exp_avg

                            # Weight decay
                            if self.defaults["weight_decay"] > 0.0:
                                p1.data.add_(p1.data, alpha=-self.defaults["lr"] * self.defaults["weight_decay"])
                            
                            p2_previous = copy.deepcopy(p2.data)

                        # Update p2
                        if p2.grad is not None:
                            state = self.state[p2]
                            if len(state) == 0:
                                state["step"] = 0
                                state["exp_avg"] = torch.zeros_like(p2.data)
                                state["exp_avg_sq"] = torch.eye(self.rank).to(p2.device)
                                state['p1_previous'] = torch.ones_like(p1.data)

                            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                            beta1, beta2 = self.defaults["betas"]
                            p1_previous = state['p1_previous']
                            state["step"] += 1

                            dim_1 = p1.data.shape[0] // 2

                            ###
                            grad2_scaled0 = p2.grad.data[0:dim_1, :]
                            c0 = p1.data[0:self.rank, :]
                            try:
                                grad2_scaled0 = grad2_scaled0 @ torch.inverse(c0 @ c0.T + self.reg * torch.eye(self.rank).to(c0.device))
                            except:
                                grad2_scaled0 = grad2_scaled0

                            ###
                            grad2_scaled1 = p2.grad.data[dim_1:, :]
                            c1 = p1.data[self.rank:, :]
                            try:
                                grad2_scaled1 = grad2_scaled1 @ torch.inverse(c1 @ c1.T + self.reg * torch.eye(self.rank).to(c1.device))
                            except:
                                grad2_scaled1 = grad2_scaled1

                            grad2_scaled = torch.cat([grad2_scaled0, grad2_scaled1])
                            assert grad2_scaled.shape == p2.grad.data.shape

                            # r r
                            AA_inverse = torch.inverse(p1.data @ p1.data.T + self.reg * torch.eye(self.rank).to(c0.device))
                            # rr rk kr --> rr
                            P_at =   p1_previous @ p1.T @ AA_inverse
                            P_at = P_at.to(c0.device)
                            # rr
                            exp_avg_sq = 0.5 * exp_avg_sq.to(c0.device)  + 0.5 * grad2_scaled.T @ grad2_scaled
                            ######################
                            #exp_avg_sq = torch.eye(self.rank).to(c0.device)
                            ######################
                            U, S, V = torch.linalg.svd(exp_avg_sq)
                            sqrt_inv_S = torch.diag(1.0 / torch.sqrt(S)) + self.reg * torch.eye(self.rank).to(c0.device)
                            exp_avg_sq_root = V @ sqrt_inv_S @ U # r r

                            S =  grad2_scaled @ exp_avg_sq_root # d r
                            exp_avg = 0.9 *  exp_avg.to(c0.device)  + 0.1 * S.to(c0.device)

                            step_size = self.defaults["lr"]
                            if self.defaults["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                bias_correction2 = 1.0 - beta2 ** state["step"]
                                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                            p2.data = p2.data - step_size * exp_avg

                            # Weight decay
                            if self.defaults["weight_decay"] > 0.0:
                                p2.data.add_(p2.data, alpha=-self.defaults["lr"] * self.defaults["weight_decay"])
                            p1_previous = copy.deepcopy(p1.data)

            i += 1

        return loss




### stardard adam however, we find historical second moment can not be added
####### cumm_first * cumm_second

class adamwr_first_order(Optimizer):
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0, correct_bias=True, rank=2, reg=1e-6):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))

        # Collect all parameters from the model
        params = list(model.parameters())
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super(adamwr_first_order, self).__init__(params, defaults)

        self.model = model
        self.rank = rank
        self.reg = reg
        print(f'{self.reg=}')

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
            if name1.endswith("lora_A.default.weight"):  # Check if the parameter name matches
                if i + 1 < len(named_params):
                    name2, p2 = named_params[i + 1]
                    i += 1  # Move to the next pair

                    # Only proceed if gradients are not None
                    if p1.grad is not None or p2.grad is not None:
                        # Update p1
                        if p1.grad is not None:
                            state = self.state[p1]
                            if len(state) == 0:
                                state["step"] = 0
                                state["exp_avg"] = torch.zeros_like(p1.data)
                                state["exp_avg_sq"] = torch.eye(self.rank).to(p1.device)
                                state['p2_previous'] = torch.ones_like(p2.data)

                            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                            beta1, beta2 = self.defaults["betas"]
                            p2_previous = state['p2_previous']
                            state["step"] += 1

                            dim_1 = p2.data.shape[0] // 2

                            ###
                            grad1_scaled0 = p1.grad.data[0:self.rank, :]
                            c0 = p2.data[0:dim_1, :]
                            try:
                                grad1_scaled0 = torch.inverse(c0.T @ c0 + self.reg * torch.eye(self.rank).to(c0.device)) @ grad1_scaled0
                            except:
                                grad1_scaled0 = grad1_scaled0

                            ###
                            grad1_scaled1 = p1.grad.data[self.rank:, :]
                            c1 = p2.data[dim_1:, :]
                            try:
                                grad1_scaled1 = torch.inverse(c1.T @ c1 + self.reg * torch.eye(self.rank).to(c1.device)) @ grad1_scaled1
                            except:
                                grad1_scaled1 = grad1_scaled1

                            grad1_scaled = torch.cat([grad1_scaled0, grad1_scaled1])
                            assert grad1_scaled.shape == p1.grad.data.shape

                            # r r
                            #BB_inverse = torch.inverse(p2.data.T @ p2.data + self.reg * torch.eye(self.rank).to(c0.device))
                            # rr rk kr --> rr
                            #P_at = BB_inverse @ p2.data.T @ p2_previous
                            #P_at = P_at.to(c0.device)
                            # rr second moment
                            exp_avg_sq = 0.9 * exp_avg_sq.to(c0.device) + 0.1 * grad1_scaled @ grad1_scaled.T
                            ######################
                            #exp_avg_sq = torch.eye(self.rank).to(c0.device)
                            ######################
                            U, S, V = torch.linalg.svd(exp_avg_sq)
                            sqrt_inv_S = torch.diag(1.0 / torch.sqrt(S)) + self.reg * torch.eye(self.rank).to(c0.device)
                            exp_avg_sq_root = V @ sqrt_inv_S @ U.T  # r r SVD???

                            #S =  exp_avg_sq_root @ grad1_scaled  # r d

                            exp_avg = 0.9 * exp_avg.to(c0.device) + 0.1 * grad1_scaled


                            step_size = self.defaults["lr"]
                            if self.defaults["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                bias_correction2 = 1.0 - beta2 ** state["step"]
                                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                            p1.data = p1.data - step_size * exp_avg_sq_root @ exp_avg

                            # Weight decay
                            if self.defaults["weight_decay"] > 0.0:
                                p1.data.add_(p1.data, alpha=-self.defaults["lr"] * self.defaults["weight_decay"])
                            
                            p2_previous = copy.deepcopy(p2.data)

                        # Update p2
                        if p2.grad is not None:
                            state = self.state[p2]
                            if len(state) == 0:
                                state["step"] = 0
                                state["exp_avg"] = torch.zeros_like(p2.data)
                                state["exp_avg_sq"] = torch.eye(self.rank).to(p2.device)
                                state['p1_previous'] = torch.ones_like(p1.data)

                            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                            beta1, beta2 = self.defaults["betas"]
                            p1_previous = state['p1_previous']
                            state["step"] += 1

                            dim_1 = p1.data.shape[0] // 2

                            ###
                            grad2_scaled0 = p2.grad.data[0:dim_1, :]
                            c0 = p1.data[0:self.rank, :]
                            try:
                                grad2_scaled0 = grad2_scaled0 @ torch.inverse(c0 @ c0.T + self.reg * torch.eye(self.rank).to(c0.device))
                            except:
                                grad2_scaled0 = grad2_scaled0

                            ###
                            grad2_scaled1 = p2.grad.data[dim_1:, :]
                            c1 = p1.data[self.rank:, :]
                            try:
                                grad2_scaled1 = grad2_scaled1 @ torch.inverse(c1 @ c1.T + self.reg * torch.eye(self.rank).to(c1.device))
                            except:
                                grad2_scaled1 = grad2_scaled1

                            grad2_scaled = torch.cat([grad2_scaled0, grad2_scaled1])
                            assert grad2_scaled.shape == p2.grad.data.shape

                            # r r
                            #A_inverse = torch.inverse(p1.data @ p1.data.T + self.reg * torch.eye(self.rank).to(c0.device))
                            # rr rk kr --> rr
                            #P_at =   p1_previous @ p1.T @ AA_inverse
                            #P_at = P_at.to(c0.device)
                            # rr
                            exp_avg_sq = 0.9 * exp_avg_sq.to(c0.device)  + 0.1 * grad2_scaled.T @ grad2_scaled
                            ######################
                            #exp_avg_sq = torch.eye(self.rank).to(c0.device)
                            ######################
                            U, S, V = torch.linalg.svd(exp_avg_sq)
                            sqrt_inv_S = torch.diag(1.0 / torch.sqrt(S)) + self.reg * torch.eye(self.rank).to(c0.device)
                            exp_avg_sq_root = V @ sqrt_inv_S @ U # r r

                            #S =  grad2_scaled @ exp_avg_sq_root # d r
                            exp_avg = 0.9 *  exp_avg.to(c0.device)  + 0.1 * grad2_scaled

                            step_size = self.defaults["lr"]
                            if self.defaults["correct_bias"]:
                                bias_correction1 = 1.0 - beta1 ** state["step"]
                                bias_correction2 = 1.0 - beta2 ** state["step"]
                                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                            p2.data = p2.data - step_size * exp_avg @ exp_avg_sq_root

                            # Weight decay
                            if self.defaults["weight_decay"] > 0.0:
                                p2.data.add_(p2.data, alpha=-self.defaults["lr"] * self.defaults["weight_decay"])
                            p1_previous = copy.deepcopy(p1.data)

            i += 1

        return loss