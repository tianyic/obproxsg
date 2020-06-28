import time
import os
import csv
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
from torch.optim.lr_scheduler import StepLR

import argparse
from datasets import Dataset
from utils import compute_F, check_accuracy
from backend import Model


def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_', default=1e-4, type=float, help='weighting parameters')
    parser.add_argument('--max_epoch', default=200, type=int)
    parser.add_argument('--backend', choices=['mobilenetv1', 'resnet18'], type=str, required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dataset_name', choices=['cifar10', 'fashion_mnist'], type=str, required=True)
    return parser.parse_args()


class OBProxSG(Optimizer):
    def __init__(self, params, lr=required, lambda_=required, epochSize=required, Np=required, No='inf', eps=0.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if lambda_ is not required and lambda_ < 0.0:
            raise ValueError("Invalid lambda: {}".format(lambda_))
        if Np is not required and Np < 0.0:
            raise ValueError("Invalid Np: {}".format(Np))
        if epochSize is not required and epochSize < 0.0:
            raise ValueError("Invalid epochSize: {}".format(epochSize))
    
        self.Np = Np
        self.No = No
        self.epochSize = epochSize
        self.step_count = 0
        self.iter = 0
        
        defaults = dict(lr=lr, lambda_=lambda_, eps=eps)
        super(OBProxSG, self).__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        No = float('inf') if self.No == 'inf' else self.No
        if self.step_count % (self.Np+No) < self.Np:
            doNp = True
            if self.iter == 0:
                print('Prox-SG Step')
        else:
            doNp = False
            if self.iter == 0:
                print('Orthant Step')
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_f = p.grad.data
                
                if doNp:
                    s = self.calculate_d(p.data, grad_f, group['lambda_'], group['lr'])
                    p.data.add_(1, s)
                else:
                    state = self.state[p]
                    if 'zeta' not in state.keys():
                        state['zeta'] = torch.zeros_like(p.data)
                    state['zeta'].zero_()
                    state['zeta'][p > 0] = 1
                    state['zeta'][p < 0] = -1
                    
                    hat_x = self.gradient_descent(p.data, grad_f, state['zeta'], group['lambda_'], group['lr'])
                    proj_x = self.project(hat_x, state['zeta'], group['eps'])
                    p.data.copy_(proj_x.data)
                        
        self.iter += 1
        if self.iter >= self.epochSize:
            self.step_count += 1
            self.iter = 0

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
    
    def gradient_descent(self, x, grad_f, zeta, lambda_, lr):
        grad = torch.zeros_like(grad_f)
        grad[zeta>0] = grad_f[zeta>0] + lambda_
        grad[zeta<0] = grad_f[zeta<0] - lambda_
        hat_x = x - lr * grad
        return hat_x
    
    def project(self, trial_x, zeta, eps):
        proj_x = torch.zeros_like(trial_x)
        keep_indexes = ( (trial_x * zeta) > eps )
        proj_x[keep_indexes] = trial_x[keep_indexes]
        return proj_x

        
def main():
    args = ParseArgs()
    lambda_ = args.lambda_
    max_epoch = args.max_epoch
    backend = args.backend
    dataset_name = args.dataset_name
    lr = 0.1
    batch_size = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainloader, testloader, num_classes = Dataset(dataset_name)
    model = Model(backend)
    model = model.to(device)
    
    weights = [w for name, w in model.named_parameters() if "weight" in name]
    num_features = sum([w.numel() for w in weights])
    num_samples = len(trainloader) * trainloader.batch_size
    
    n = num_features
    m = num_samples

    criterion = nn.CrossEntropyLoss()
    optimizer = OBProxSG(model.parameters(), lr=lr, lambda_=lambda_, epochSize=len(trainloader), Np=int(max_epoch/10))
    scheduler = StepLR(optimizer, step_size=60, gamma=0.1)

    os.makedirs('results', exist_ok=True)
    setting = 'obproxsg_plus_%s_%s_%E'%(backend, dataset_name, lambda_)
    csvname = os.path.join('results', setting + '.csv')
    print('The csv file is %s'%csvname)

    csvfile = open(csvname, 'w', newline='')
    fieldnames = ['epoch', 'F_value', 'f_value', 'norm_l1_x', 'density', 'validation_acc', 'train_time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=",")
    writer.writeheader()

    alg_start_time = time.time()

    epoch = 0

    while True:
        epoch_start_time = time.time()
        if epoch >= max_epoch:
            break
            
        for index, (X, y) in enumerate(trainloader):
            X = X.to(device)
            y = y.to(device)
            y_pred = model.forward(X)

            f = criterion(y_pred, y)
            optimizer.zero_grad()
            f.backward()
            optimizer.step()

        scheduler.step()
        epoch += 1

        train_time = time.time() - epoch_start_time
        F, f, norm_l1_x = compute_F(trainloader, model, weights, criterion, lambda_)
        density = sum([torch.sum(w != 0).item() for w in weights]) / num_features
        accuracy = check_accuracy(model, testloader)
        writer.writerow({'epoch': epoch, 'F_value': F, 'f_value': f, 'norm_l1_x': norm_l1_x, 'density': density, 'validation_acc': accuracy, 'train_time': train_time})
        csvfile.flush()
        print("Epoch {}: {:2f}seconds ...".format(epoch, train_time))


    alg_time = time.time() - alg_start_time
    writer.writerow({'train_time': alg_time / epoch})

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model, os.path.join('checkpoints', setting + '.pt'))
    csvfile.close()
    

if __name__ == "__main__":
    main()
