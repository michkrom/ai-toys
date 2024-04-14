# torch utils
#
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# Generic data feeder - Dataset is list of pairs X --> y
class DatasetFeeder(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, y = self.data[idx]
        return X, y


class NNet(nn.Module):
    """multi-layer NN with input_dim and layers specified as [(dim, activation)]"""

    def __init__(self, input_dim, layers):
        super(NNet, self).__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            dim, act = (layer, None) if isinstance(layer, int) else layer
            mod = nn.Linear(input_dim, dim, bias=True)
            #      nn.init.kaiming_uniform_(mod.weight)
            nn.init.xavier_uniform_(mod.weight)
            self.add_module(f"fc{i}", mod)
            self.layers.append((mod, act))
            input_dim = dim

    def forward(self, x):
        for transfom, activation in self.layers:
            x = transfom(x)
            if activation is not None:
                x = activation(x)
        return x

    def train(
        self,
        train_data,
        epochs,
        learning_rate,
        optimizer=None,
        criterion=None,
        progress_reporter=None,
    ):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if criterion is None:
            criterion = nn.MSELoss()
        if progress_reporter is None:
            progress_reporter = lambda epoch, loss: print(
                f"Epoch: {epoch}, Loss: {loss:.4f}"
            )

        train_dataset = DatasetFeeder(train_data)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        train_loss_values = []

        for epoch in range(epochs):
            for input, output in train_loader:
                # Forward pass
                outputs = self(input)
                loss = criterion(outputs, output)
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss_values.append((epoch, loss.item))
            progress_reporter(epoch + 1, loss.item())

        self.train_loss_values = train_loss_values

    def visualize_training():
        import matplotlib.pyplot as plt

        step = np.linspace(0, 100, 10500)
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.plot(step, np.array(loss_values))
        plt.title("Step-wise Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
