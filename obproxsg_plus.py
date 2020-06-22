import time
import os
import csv
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required

import argparse
from datasets import Dataset
from utils import compute_F, check_accuracy
from backend import Model

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmbda', default=1e-4, type=float, help='weighting parameters')
    parser.add_argument('--max_epoch', default=200, type=int)
    parser.add_argument('--backend', choices=['mobilenetv1', 'resnet18'], type=str, required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dataset_name', choices=['cifar10', 'fashion_mnist'], type=str, required=True)
    return parser.parse_args()

class OBProxSG(Optimizer):
    def __init__(self, params, alpha=required, lmbda = required):
        if alpha is not required and alpha < 0.0:
            raise ValueError("Invalid learning rate: {}".format(alpha))

        if lmbda is not required and lmbda < 0.0:
            raise ValueError("Invalid lambda: {}".format(lmbda))

        defaults = dict(alpha=alpha, lmbda=lmbda)
        super(OBProxSG, self).__init__(params, defaults)

    def calculate_d(self, x, grad_f, lmbda, alpha):
        '''
            Calculate d for Omega(x) = ||x||_1
        '''
        trial_x  = torch.zeros(x.shape).to(device)
        pos_shrink = x - alpha * grad_f - alpha * lmbda
        neg_shrink = x - alpha * grad_f + alpha * lmbda
        pos_shrink_idx = (pos_shrink > 0)
        neg_shrink_idx = (neg_shrink < 0)
        trial_x[pos_shrink_idx] = pos_shrink[pos_shrink_idx]
        trial_x[neg_shrink_idx] = neg_shrink[neg_shrink_idx]
        d = trial_x - x

        return d
    
    def __setstate__(self, state):
        super(OBProxSG, self).__setstate__(state)

    def sprox_step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad_f = p.grad.data
                
                if len(p.shape) > 1:
                    s = self.calculate_d(p.data, grad_f, group['lmbda'], group['alpha'])
                    p.data.add_(1, s)
                else:
                    p.data.add_(-group['alpha'], grad_f)
                    
        return loss
    
    
    def gradient_descent(self, x, grad_f, zeta, lmbda, alpha):
        '''
            GRADIENT_DESCENT Summary of this function goes here
        '''
        grad = torch.zeros_like(grad_f)
        grad[zeta>0] = grad_f[zeta>0] + lmbda
        grad[zeta<0] = grad_f[zeta<0] - lmbda
        hat_x = x - alpha * grad
        return hat_x
    
    def project(self, trial_x, zeta):
        proj_x = torch.zeros_like(trial_x)
        keep_indexes = ( (trial_x * zeta) > 0 )
        proj_x[keep_indexes] = trial_x[keep_indexes]
        return proj_x
    
    def ow_step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if p.grad is None:
                    continue
                grad_f = p.grad.data
                
                if len(p.shape) > 1:
                    hat_x = self.gradient_descent(p.data, grad_f, state['zeta'], group['lmbda'], group['alpha'])
                    proj_x = self.project(hat_x, state['zeta'])
                    p.data.copy_(proj_x.data)

                else:
                    p.data.add_(-group['alpha'], grad_f)

                    
        return loss     
    
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
        for group in self.param_groups:
            group['alpha'] = group['alpha'] / 10
        print('lr:', self.param_groups[0]['alpha'])
    
    def init_zeta(self):
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'zeta' not in state.keys():
                    state['zeta'] = torch.zeros_like(p.data)
                state['zeta'].zero_()
                state['zeta'][p > 0] = 1
                state['zeta'][p < 0] = -1

if __name__ == "__main__":
    
    args = ParseArgs()
    lmbda = args.lmbda
    max_epoch = args.max_epoch
    backend = args.backend
    dataset_name = args.dataset_name
    alpha = 0.1
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
    optimizer = OBProxSG(model.parameters(), alpha=alpha, lmbda=lmbda)

    os.makedirs('results', exist_ok=True)
    setting = 'obproxsg_plus_%s_%s_%E'%(backend, dataset_name, lmbda)
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

        if epoch < 100:          
            print('Prox-SG Step')
            for index, (X, y) in enumerate(trainloader):
                X = X.to(device)
                y = y.to(device)
                y_pred = model.forward(X)

                f = criterion(y_pred, y)
                optimizer.zero_grad()
                f.backward()
                optimizer.sprox_step()

        else:
            print('Orthant Step')
            for index, (X, y) in enumerate(trainloader):
                optimizer.init_zeta()
                X = X.to(device)
                y = y.to(device)

                y_pred = model.forward(X)
                f = criterion(y_pred, y)
                optimizer.zero_grad()
                f.backward()
                optimizer.ow_step()


        epoch += 1
        if epoch in [75, 130, 180]:
            optimizer.adjust_learning_rate(epoch)

        train_time = time.time() - epoch_start_time
        F, f, norm_l1_x = compute_F(trainloader, model, weights, criterion, lmbda)
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

