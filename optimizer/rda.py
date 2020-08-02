import torch
import math
from torch.optim.optimizer import Optimizer, required


class RDA(Optimizer):
    def __init__(self, params, lambda_=required, gamma=required, epochSize=required):
        if lambda_ is not required and lambda_ < 0.0:
            raise ValueError("Invalid lambda: {}".format(lambda_))

        if gamma is not required and gamma < 0.0:
            raise ValueError("Invalid gamma: {}".format(gamma))

        if epochSize is not required and epochSize < 0.0:
            raise ValueError("Invalid epochSize: {}".format(epochSize))

        self.init_run = True
        self.epochSize = epochSize
        self.iter = 0
        defaults = dict(lambda_=lambda_, gamma=gamma)
        super(RDA, self).__init__(params, defaults)

    def shrink(self, i, bar_g, lambda_, gamma):
        '''
            Soft l1 shrinkage operator for RDA
        '''
        x = torch.zeros_like(bar_g)
        mask = abs(bar_g) > lambda_
        x[mask] = -math.sqrt(i)/gamma*(bar_g[mask] -
                                       lambda_*torch.sign(bar_g[mask]))
        return x

    def step(self, closure=None):
        if self.init_run:
            self.reset_state()
            self.init_run = False

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad_f = p.grad.data

                state = self.state[p]

                state['i'] += 1
                lambda_ = group['lambda_']
                gamma = group['gamma']

                i = state['i']
                state['bar_g'].mul_((i-1)/i).add_(1 / i, grad_f)

                x = torch.zeros_like(p.data)
                mask = abs(state['w_0'] - math.sqrt(i) / gamma *
                           state['bar_g']) > math.sqrt(i) / gamma * lambda_

                if len(p.shape) > 1:  # weights
                    x[mask] = state['w_0'][mask] - math.sqrt(i)/gamma*(state['bar_g'][mask] + lambda_*torch.sign(
                        state['w_0'][mask] - math.sqrt(i) / gamma * state['bar_g'][mask]))
                    p.data.copy_(x)
                else:
                    x[mask] = state['w_0'][mask] - \
                        math.sqrt(i)/gamma*(state['bar_g'][mask])
                    p.data.copy_(x)

                print(state['i'], end=';')

        self.iter += 1
        if self.iter >= self.epochSize:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    state['i'] = 0
            self.iter = 0


        return loss

    def reset_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['i'] = 0

                if 'bar_g' not in state.keys():
                    state['bar_g'] = torch.zeros_like(p.data)
                state['bar_g'].zero_()

                if 'w_0' not in state.keys():
                    state['w_0'] = torch.zeros_like(p.data)
                state['w_0'].copy_(p.data)
