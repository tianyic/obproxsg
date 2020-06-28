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
from obproxsg_plus import OBProxSG


def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_', default=1e-4, type=float, help='weighting parameters')
    parser.add_argument('--max_epoch', default=200, type=int)
    parser.add_argument('--backend', choices=['mobilenetv1', 'resnet18'], type=str, required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dataset_name', choices=['cifar10', 'fashion_mnist'], type=str, required=True)
    return parser.parse_args()


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
    optimizer = OBProxSG(model.parameters(), lr=lr, lambda_=lambda_, epochSize=len(trainloader), Np=5, No=5)
    scheduler = StepLR(optimizer, step_size=60, gamma=0.1)

    os.makedirs('results', exist_ok=True)
    setting = 'obproxsg_%s_%s_%E'%(backend, dataset_name, lambda_)
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

