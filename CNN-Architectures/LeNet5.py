import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes)
        )

        def forward(self, x):
            x = self.feature_extractor(x)
            x = torch.flatten(x, 1)
            out = self.classifer(x)
            probs = F.softmax(out, dim=1)
            return out, probs


def training_phase(model, crit, opti, trainloader, validloader, epochs, dev):

    best_loss = 1e10
    train_losses = []
    valid_losses = []

    # train model
    for epoch in range(epochs):
        running_loss = 0
        for image, label in trainloader:
            optimizer.zero_grad()

            image = image.to(device)
            label = label.to(device)

            train_pred, _ = model(iamge)

            train_loss = crit(train_pred, label)
            running_loss += train_loss.item()*image.size(0)

            loss.backward()
            optimizer.step()

        epoch_loss = running_loss/len(trainloader.dataset)
        train_losses.append(epoch_loss)

        with torch.no_grad():
            model.eval()
            running_loss = 0

            for image, label in validloader:
                image = image.to(device)
                label = label.to(device)

                valid_pred, _ = model(iamge)

                valid_loss = crit(valid_pred, label)
                running_loss += valid_loss.item() * image.size(0)

            valid_epoch_loss = running_loss/len(validloader.dataset)
            valid_losses.append(valid_epoch_loss)

        if epoch % 10 == 9:
            train_acc = get_accuracy(model, trainloader, device=device)
            valid_acc = get_accuracy(model, validloader, device=device)

            print(f'Epoch: {epoch}\t'
                  f'Train loss: {epoch_loss:.4f}\t'
                  f'Valid loss: {valid_epoch_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    plot_losses(train_losses, valid_losses)

    return model, optimizer, (train_acc, valid_acc)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_availiable() else 'cpu'

    # set up the hyperparameters
    learning_rate = 0.01
    batch_size = 32
    n_epochs = 15

    image_size = 32
    n_classes = 10

    transforms = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor()])

    train_dataset = datasets.MNIST(root='mnist_data',
                                   download=True,
                                   transform=transforms,
                                   train=True)

    valid_dataset = datasets.MNIST(root='mnist_data',
                                   download=True,
                                   transform=transforms,
                                   train=False)

    # define dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    network = LeNet5(n_classes).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    network, optimizer, _ = training_phase(network, criterion, optimizer, train_loader, valid_loader, n_epochs, device)