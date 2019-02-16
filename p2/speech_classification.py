import torch
import numpy as np
import torch.nn as nn
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from wsj_loader import WSJ


class WSJDataset(Dataset):
    def __init__(self, dataset, context_size=12):
        # TODO: remove indexing after compiling successfully
        self.X = dataset[0][0:100]
        self.Y = dataset[1]
        self.is_test = False
        if self.Y is None:
            self.is_test = True
        else:
            self.Y = self.Y[0:100]
            assert(self.X.shape[0] == self.Y.shape[0])

        self.context_size = context_size
        # flattened features that concatenate nearby k frames
        self.concat_X = []
        self.concat_Y = []
        self.feature_generation()

    def feature_generation(self):
        # empty previous data
        self.concat_X = []
        self.concat_Y = []

        for i, frames in enumerate(self.X):
            # zero padding to shape of [(2k+len(frames)) * 40]
            padded = np.pad(frames, ((self.context_size, self.context_size), (0, 0)), 'constant', constant_values=0)
            for frame_id, frame in enumerate(frames):
                neighbor_frames = padded[frame_id:frame_id + 2 * self.context_size + 1, :]
                self.concat_X.append(neighbor_frames)
                if not self.is_test:
                    self.concat_Y.append(self.Y[i][frame_id])
        assert(self.is_test or (not self.is_test and len(self.concat_X) == len(self.concat_Y)))

    def __len__(self):
        return len(self.concat_X)

    def __getitem__(self, item):
        x = self.concat_X[item]
        label = self.concat_Y[item]
        return x, label

    # def frames_concatenation(self, frames, frame_id):
    #     # zero padding to shape of [(2k+len(frames)) * 40]
    #     padded = np.pad(frames, ((self.context_size, self.context_size), (0, 0)), 'constant', constant_values=0)
    #     return padded[frame_id:frame_id + 2 * self.context_size + 1, :]


""" MLP Network class definition """


class MLPNetwork(nn.Module):
    # you can use the layer sizes as initialization arguments if you want to
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Baseline layer structure:  [input_size, 1024, 1024, 512, 512, output_size]
        layer_size = [input_size] + hidden_size + [output_size]
        self.layers = []
        for i in range(len(layer_size) - 1):
            self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i != len(layer_size) - 2:
                self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(layer_size[i + 1], momentum=0.1))
        self.model = nn.Sequential(*self.layers)

    def forward(self, input):
        self.input = input
        self.output = self.model(self.input)
        return self.output


""" function for training """


def train(net, optimizer, criterion, train_loader, gpu, epoch):
    net.train()
    device = torch.device("cuda" if gpu else "cpu")
    net.to(device)
    for batch_idx, (inputs, labels) in train_loader:
        # cuda
        inputs = torch.tensor(inputs.to(device), requires_grad=True)
        labels = torch.tensor(labels.to(device), requires_grad=True)
        # mini-batch GD
        optimizer.zero_grad()
        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
            print("[Train Epoch %d] batch_idx=%d, loss=%.4f", epoch, batch_idx, loss.item())


""" function for validation """


def validate(net, criterion, dev_loader, gpu, epoch):
    net.eval()
    device = torch.device("cuda" if gpu else "cpu")
    net.to(device)

    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
    for inputs, labels in dev_loader:
        # cuda
        inputs = torch.tensor(inputs.to(device), requires_grad=True)
        labels = torch.tensor(labels.to(device), requires_grad=True)
        # compute result and evaluate loss
        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        # predict labels
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == outputs).sum().item()

    running_loss /= len(dev_loader.dataset)
    acc = (correct_predictions / total_predictions) * 100.0
    print("[Val Epoch %d] loss=%.4f, acc=%.4f", epoch, running_loss, acc)


""" function for testing """


def test(net, test_loader):
    net.eval()
    output = net.forward()



""" main routine """


def main():
    # parameters
    gpu = False
    gpu = gpu and torch.cuda.is_available()
    epochs = 6
    lr = 1e-3
    batch_size = 10
    context_size = 12
    params = {"batch_size": batch_size, "shuffle": True, "num_workers": 1, "pin_memory": gpu}

    input_size = (2 * context_size + 1) * 40
    output_size = 138

    # load data
    print("loading WSJ...")
    loader = WSJ()
    print("loading training data...")
    train_loader = DataLoader(WSJDataset(loader.train, context_size=context_size), **params)
    print("loading dev data...")
    dev_loader = DataLoader(WSJDataset(loader.dev, context_size=context_size), **params)
    print("loading test data...")
    test_loader = DataLoader(WSJDataset(loader.test, context_size=context_size), **params)

    # model definition
    net = MLPNetwork(input_size, [1024, 1024, 512, 512], output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # train for epochs
    for epoch in range(epochs):
        print("===========Training==========")
        train(net, optimizer, criterion, train_loader, gpu, epoch)
        print("==========Validation=========")
        validate(net, criterion, dev_loader, gpu, epoch)

    print("Congratulations! Training completed successfully!")


if __name__ == "__main__":
    main()
