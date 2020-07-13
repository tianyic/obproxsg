import torch


def compute_F(trainloader, model, weights, criterion, lmbda):
    f = 0.0
    device = next(model.parameters()).device
    for index, (X, y) in enumerate(trainloader):
        X = X.to(device)
        y = y.to(device)
        y_pred = model.forward(X)
        f1 = criterion(y_pred, y)
        f += float(f1)
    f /= len(trainloader)
    norm_l1_x_list = []
    for w in weights:
        norm_l1_x_list.append(torch.norm(w, 1).item())
    norm_l1_x = sum(norm_l1_x_list)
    F = f + lmbda * norm_l1_x

    return F, f, norm_l1_x


def check_accuracy(model, testloader):
    correct = 0
    total = 0
    model = model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for (X, y) in testloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    model = model.train()
    accuracy = correct / total
    return accuracy
