import torch
import cv2
import pandas as pd
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

header = ['img_path', 'class']
dataset = pd.read_csv('dataset.csv', names=header)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'ba', 'pa')
images = []
labels = []
for i in range(len(dataset)):
    img = cv2.imread(dataset.iloc[i, 0])
    label = dataset.iloc[i, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
    img = img.reshape(1, 28, 28)
    images.append(img/255)
    labels.append(label)


train_data, test_data, train_label, test_label = train_test_split(
    images, labels, shuffle=True, random_state=10, train_size=0.9)
train_data = torch.tensor(train_data).float()
test_data = torch.tensor(test_data).float()
train_label = torch.tensor(train_label)
test_label = torch.tensor(test_label)

batch_size = 10
train = TensorDataset(train_data, train_label)
test = TensorDataset(test_data, test_label)
train_loader = DataLoader(train, batch_size=batch_size)
test_loader = DataLoader(test, batch_size=batch_size)


class ConvNeural(nn.Module):
    def __init__(self):
        super(ConvNeural, self).__init__()
        self.forward_propogation = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(32*5*5, 16, bias=True),
            nn.ReLU(),

            nn.Linear(16, 12, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.forward_propogation(x)
        return output


model = ConvNeural()

learning_rate = 0.13
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train(data_dataloader, model, loss_fn, optimizer):
    size = len(data_dataloader.dataset)
    lossess = []
    accuracies = []
    for batch, (x, y) in enumerate(data_dataloader):
        pred = model(x)
        loss = loss_fn(pred, y)
        cat = torch.argmax(pred, dim=1)
        accuracy = (cat == y).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(
                f"Loss: {loss:>7f}   [{current:>5d}/{size:>5d}] Accuracy: {accuracy * 100:>7f}")
            lossess.append(loss)
            accuracies.append(accuracy)
    return lossess, accuracies


def test(data_dataloader, model):
    size = len(data_dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (x, y) in data_dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(
        f"\n Test Error: \n Avg loss: {test_loss:>8f} Test Accuracy:{correct * 100:>5f}% \n")


epochs = 8
losses, accuracies = [], []
for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------------------")
    loss, accuracy = train(train_loader, model, loss_fn, optimizer)
    for l, a in zip(loss, accuracy):
        losses.append(l)
        accuracies.append(a)
    test(test_loader, model)

torch.save(model.state_dict(), 'model.pth')
print("Saved model")
