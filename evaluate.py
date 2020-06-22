import argparse
import torch
import torch.nn as nn

from datasets import Dataset
from utils import compute_F, check_accuracy

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['mobilenetv1', 'resnet18'], type=str, required=True)
    parser.add_argument('--lmbda', default=1e-4, type=float, help='weighting parameters')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--ckpt', required=True, type=str)
    parser.add_argument('--dataset_name', choices=['cifar10', 'fashion_mnist'], type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    
    args = ParseArgs()
    backend = args.backend
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    lmbda = args.lmbda

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainloader, testloader, num_classes = Dataset(dataset_name)
    model = torch.load(args.ckpt).to(device)
    
    weights = [w for name, w in model.named_parameters() if "weight" in name]
    num_features = sum([w.numel() for w in weights])
    
    n = num_features

    criterion = torch.nn.CrossEntropyLoss()

    F, f, norm_l1_x = compute_F(trainloader, model, weights, criterion, lmbda)
    density = sum([torch.sum(w != 0).item() for w in weights]) / num_features
    accuracy = check_accuracy(model, testloader)


    print('F:', F)
    print('f:', f)
    print('density:', density)
    print('validation accuracy:', accuracy)
