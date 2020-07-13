from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, FashionMNIST
import torchvision.transforms as transforms

DATA_DIR = "datasets"


def Dataset(dataset_name, batch_size=128):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if dataset_name == 'cifar10':
        print('Dataset: CIFAR10.')
        trainset = CIFAR10(root=DATA_DIR, train=True, download=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize]))

        testset = CIFAR10(root=DATA_DIR, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize]))
        num_classes = 10
    elif dataset_name == 'fashion_mnist':
        print('Dataset: FashionMNIST.')
        trainset = FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize]))

        testset = FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize]))
        num_classes = 10
    else:
        raise ValueError

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    return trainloader, testloader, num_classes
