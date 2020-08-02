import torch
from torch.optim.optimizer import Optimizer, required

class ProxSVRG(Optimizer):
    def __init__(self, params, lr=required, lambda_=required, epochSize=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid lr: {}".format(lr))
        
        if lambda_ is not required and lambda_ < 0.0:
            raise ValueError("Invalid lambda: {}".format(lambda_))

        if epochSize is not required and epochSize < 0.0:
            raise ValueError("Invalid epochSize: {}".format(epochSize))

        self.epochSize = epochSize
            
        defaults = dict(lr=lr, lambda_=lambda_)
        super(ProxSVRG, self).__init__(params, defaults)
    
    def calculate_d(self, x, grad_f, lambda_, lr):
        trial_x  = torch.zeros_like(x)
        pos_shrink = x - lr * grad_f - lr * lambda_
        neg_shrink = x - lr * grad_f + lr * lambda_
        pos_shrink_idx = (pos_shrink > 0)
        neg_shrink_idx = (neg_shrink < 0)
        trial_x[pos_shrink_idx] = pos_shrink[pos_shrink_idx]
        trial_x[neg_shrink_idx] = neg_shrink[neg_shrink_idx]
        d = trial_x - x

        return d

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad_f = p.grad.data
                state = self.state[p]

                p.data.copy_( state['xs_sum'] / state['i'] )
        return loss
    

    def init_epoch(self):
        '''
            Revoked at the begining of the epoch
        '''
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['i'] = 0
                if 'hat_v' not in state.keys():
                    state['hat_v'] = torch.zeros_like(p.grad.data)
                state['hat_v'].copy_(p.grad.data)
                state['hat_v'].div_(self.epochSize)

                if 'hat_x' not in state.keys():
                    state['hat_x'] = torch.zeros_like(p.data)
                state['hat_x'].copy_(p.data)

                if 'xs_end' not in state.keys():
                    state['xs_end'] = torch.zeros_like(state['hat_x'])
                state['xs_end'].copy_(state['hat_x'])
                if 'xs_sum' not in state.keys():
                    state['xs_sum'] = torch.zeros_like(state['hat_x'])
                state['xs_sum'].zero_()

    def set_weights_as_xs(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                p.data.copy_(state['xs_end'])

    def set_weights_as_hat_x(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                p.data.copy_(state['hat_x'])


    def save_grad_f(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'grad_f' not in state.keys():
                    state['grad_f'] = torch.zeros_like(p.grad.data)
                state['grad_f'].copy_(p.grad.data)

    def save_grad_f_hat(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'grad_f_hat' not in state.keys():
                    state['grad_f_hat'] = torch.zeros_like(p.grad.data)
                state['grad_f_hat'].copy_(p.grad.data)


    def update_xs(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['i'] += 1
                v = state['grad_f'] - state['grad_f_hat'] + state['hat_v']
                if len(p.shape) > 1: # weights
                    s = self.calculate_d(state['xs_end'], v, group['lambda_'], group['lr'])
                    state['xs_end'].add_(s)
                    state['xs_sum'].add_(state['xs_end'])
                else:
                    state['xs_end'].add_(-group['lr'], v)
                    state['xs_sum'].add_(state['xs_end'])
                        
                
