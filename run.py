import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from datasets import Dataset
from model.mobilenetv1 import MobileNet
from model.resnet import ResNet18
from optimizer import *
from utils import check_accuracy, compute_F


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_', default=1e-4,
                        type=float, help='weighting parameters')
    parser.add_argument('--max_epoch', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--learning_rate', '-lr', default=0.1, type=float)
    parser.add_argument(
        '--model', choices=['mobilenetv1', 'resnet18'], type=str, required=True)
    parser.add_argument(
        '--dataset_name', choices=['cifar10', 'fashion_mnist'], type=str, required=True)
    parser.add_argument('--optimizer', choices=[
                        'obproxsg_plus', 'obproxsg', 'proxsg', 'proxsvrg', 'rda'], type=str, required=True)
    return parser.parse_args()


def selectModel(modelName):
    if modelName == 'resnet18':
        print('modelName: ResNet18')
        model = ResNet18()
    elif modelName == 'mobilenetv1':
        print('modelName: MobileNetV1')
        model = MobileNet()
    else:
        raise ValueError
    return model


def main():
    args = parseArgs()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainloader, testloader, num_classes = Dataset(
        args.dataset_name, args.batch_size)
    model = selectModel(args.model)
    model = model.to(device)

    weights = [w for name, w in model.named_parameters() if "weight" in name]
    num_features = sum([w.numel() for w in weights])
    num_samples = len(trainloader) * trainloader.batch_size

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'obproxsg':
        optimizer = OBProxSG(model.parameters(), lr=args.learning_rate,
                             lambda_=args.lambda_, epochSize=len(trainloader), Np=5, No=5)
    elif args.optimizer == 'obproxsg_plus':
        optimizer = OBProxSG(model.parameters(), lr=args.learning_rate,
                             lambda_=args.lambda_, epochSize=len(trainloader), Np=int(args.max_epoch/10))
    elif args.optimizer == 'proxsg':
        optimizer = ProxSG(model.parameters(),
                           lr=args.learning_rate, lambda_=args.lambda_)
    elif args.optimizer == 'proxsvrg':
        pass
    elif args.optimizer == 'rda':
        optimizer = RDA(model.parameters(), lr=args.learning_rate, gamma=20)

    scheduler = StepLR(optimizer, step_size=60, gamma=0.1)

    os.makedirs('results', exist_ok=True)
    setting = '%s_%s_%s_%E' % (
        args.optimizer, args.model, args.dataset_name, args.lambda_)
    csvname = os.path.join('results', setting + '.csv')
    print('Results are saving to the CSV file: %s.' % csvname)

    csvfile = open(csvname, 'w', newline='')
    fieldnames = ['epoch', 'F_value', 'f_value', 'norm_l1_x',
                  'density', 'validation_acc', 'train_time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=",")
    writer.writeheader()

    alg_start_time = time.time()

    epoch = 0
    while True:
        epoch_start_time = time.time()
        if epoch >= args.max_epoch:
            break

        for _, (X, y) in enumerate(trainloader):
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
        F, f, norm_l1_x = compute_F(
            trainloader, model, weights, criterion, args.lambda_)
        density = sum([torch.sum(w != 0).item()
                       for w in weights]) / num_features
        accuracy = check_accuracy(model, testloader)
        writer.writerow({'epoch': epoch, 'F_value': F, 'f_value': f, 'norm_l1_x': norm_l1_x,
                         'density': density, 'validation_acc': accuracy, 'train_time': train_time})
        csvfile.flush()
        print("Epoch {}: {:2f}seconds ...".format(epoch, train_time))

    alg_time = time.time() - alg_start_time
    writer.writerow({'train_time': alg_time / epoch})

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model, os.path.join('checkpoints', setting + '.pt'))
    csvfile.close()


if __name__ == "__main__":
    main()
