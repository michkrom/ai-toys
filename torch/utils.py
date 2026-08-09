# torch utils
#
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
        for transform, activation in self.layers:
            x = transform(x)
            if activation is not None:
                x = activation(x)
        return x

    def train(
        self,
        train_data=None,
        epochs=None,
        learning_rate=None,
        optimizer=None,
        criterion=None,
        progress_reporter=None,
        device=None,
        converge_at=None,
    ):
        # Our signature shadows nn.Module.train(mode). Support the base-class
        # call style so containers can toggle mode / eval() without breaking.
        if isinstance(train_data, bool) and epochs is None and learning_rate is None:
            return super().train(train_data)
        if train_data is None:
            raise TypeError("NNet.train() requires train_data")
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if criterion is None:
            criterion = nn.MSELoss()
        if progress_reporter is None:
            progress_reporter = lambda epoch, loss: print(
                f"Epoch: {epoch}, Loss: {loss:.6f}"
            )

        train_dataset = DatasetFeeder(train_data)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        train_loss_values = []
        converged_epoch = None
        num_samples = len(train_dataset)

        for epoch in range(epochs):
            total = 0.0
            for input, output in train_loader:
                if device is not None:
                    input, output = input.to(device), output.to(device)
                # Forward pass
                outputs = self(input)
                loss = criterion(outputs, output)
                total += loss.item() * len(input)
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            mean_loss = total / num_samples
            train_loss_values.append((epoch, mean_loss))
            progress_reporter(epoch + 1, mean_loss)
            if converge_at is not None and mean_loss <= converge_at:
                converged_epoch = epoch + 1
                break

        self.train_loss_values = train_loss_values
        self.converged_at_epoch = converged_epoch
        return converged_epoch

    def visualize_training(self):
        import matplotlib.pyplot as plt

        epochs, losses = zip(*self.train_loss_values)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
