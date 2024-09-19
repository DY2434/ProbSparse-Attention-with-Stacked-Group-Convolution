import numpy as np
import torch
import torch.nn as nn
import argparse
from models.util import load_data_n_model
from sklearn.metrics import f1_score
from thop import profile


def calculate_f1(y_true, y_pred):
    return f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_complexity(model, input_size):
    device = next(model.parameters()).device
    input = torch.randn(1, *input_size).to(device)
    flops, params = profile(model, inputs=(input,))
    return flops, params


def train(model, tensor_loader, num_epochs, learning_rate, criterion, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_f1 = 0
        for inputs, labels in tensor_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1)
            epoch_accuracy += (predict_y == labels).float().mean().item()
            epoch_f1 += calculate_f1(labels, predict_y)

        epoch_loss /= len(tensor_loader.dataset)
        epoch_accuracy /= len(tensor_loader)
        epoch_f1 /= len(tensor_loader)
        print(f'Epoch:{epoch + 1}, Accuracy:{epoch_accuracy:.4f}, F1:{epoch_f1:.4f}, Loss:{epoch_loss:.9f}')


def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    test_f1 = 0
    with torch.no_grad():
        for inputs, labels in tensor_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predict_y = torch.argmax(outputs, dim=1)
            accuracy = (predict_y == labels).float().mean().item()
            test_acc += accuracy
            test_loss += loss.item() * inputs.size(0)
            test_f1 += calculate_f1(labels, predict_y)

    test_acc /= len(tensor_loader)
    test_loss /= len(tensor_loader.dataset)
    test_f1 /= len(tensor_loader)
    print(f"Validation accuracy:{test_acc:.4f}, F1:{test_f1:.4f}, loss:{test_loss:.5f}")


def main():
    root = '/kaggle/input/ntu-fi-har/'
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices=['UT_HAR_data', 'NTU-Fi-HumanID', 'NTU-Fi_HAR'])
    parser.add_argument('--model', choices=['ResNet18', 'ResNet50', 'GRU', 'BiLSTM', 'ViT'])
    args = parser.parse_args()

    print(f"using dataset: {args.dataset}")
    print(f"using model: {args.model}")

    train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params}")

    input_size = (3, 114, 2000)
    flops, _ = estimate_complexity(model, input_size)
    print(f"Estimated FLOPs: {flops}")

    train(
        model=model,
        tensor_loader=train_loader,
        num_epochs=train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device
    )
    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device
    )


if __name__ == "__main__":
    main()