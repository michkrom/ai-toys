# NN to predict a parity bit over 7 bit byte
#
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils import NNet


def next_byte_rand():
    return random.randint(0, 127)


byte = 0


def next_byte_lin():
    global byte
    byte = byte + 1 if byte < 127 else 0
    return byte


def bit_vec(byte, bits=7):
    return [(byte >> i) & 1 for i in range(bits - 1, -1, -1)]


# generates prity data: tensor[7] (bits) --> tensor[2] (parity bit probability 0: [0] 1: [1])
def generate_parity_data(num_samples, next_byte=next_byte_rand):
    data = []
    for _ in range(num_samples):
        data_bits = bit_vec(next_byte())
        data_tensor = torch.tensor(data_bits, dtype=torch.float32)
        parity = sum(data_bits) % 2
        # probability distribution (one-hot encoded)
        parity_probs = torch.tensor(
            [0.5, 0.5], dtype=torch.float32
        )  # Initial probabilities (using 0 is much worse)
        parity_probs[parity] = 1.0  # Set probability to 1.0 for the actual parity
        data.append((data_tensor, parity_probs))
    return data


def test_inference(model, num_samples=128):
    correct = 0
    total = 0

    for _ in range(num_samples):
        data_bits = bit_vec(next_byte_lin())
        data_tensor = torch.tensor(data_bits, dtype=torch.float32)
        actual_parity = sum(data_bits) % 2

        # Forward pass
        outputs = model(data_tensor)
        # print(data_tensor, outputs)

        # Get the predicted parity (index with highest probability)
        predicted_parity = torch.argmax(outputs, dim=0).item()

        # Check if prediction matches actual parity
        if predicted_parity == actual_parity:
            correct += 1

        total += 1

    accuracy = correct / total * 100
    return accuracy, correct, total


def signature(data):
    import hashlib

    hash = hashlib.md5()
    for bits, par in data:
        v = 0
        for bit in bits:
            v = v * 2
            v = v + int(bit)
        p = torch.argmax(par, dim=0).item()
        v = v + (p * 128)
        hash.update(bytes([v]))
    return hash.hexdigest()


########################################################################################

# for repetability
# these seeds end up training to 100% (empirical and both needed)
random.seed(31415926535897932)  # drives randomness of training data
# torch.manual_seed(42) # drives randomness for weight init

# Chasing best models (comments hold for seeded (above) runs)
# Input dimension of 7 data bits --> inner 7 --> output 2 (probability of 0 and 1)
# not converging with MSELoss but was in 90% with BCL
# model = NNet(7, ((7, nn.Tanh()), 2))
# single layer does not work better gets to mid 99% (orscilates in training in the tail)
# model = NNet(7, ((128, nn.Sigmoid()), 2))
# great - get to 100% in 22 epochs (though still depends on seeds)
# these seems to train to 100% most of the times (per random weight init)
# model = NNet(7, ((128, nn.Sigmoid()), (32, nn.Sigmoid()), 2))
# model = NNet(7, ((64, nn.Sigmoid()), (16, nn.Sigmoid()), 2))
# model = NNet(7, ((32, nn.Sigmoid()), (16, nn.Sigmoid()), 2))
# model = NNet(7, ((32, nn.Sigmoid()), (8, nn.Sigmoid()), 2))
# model = NNet(7, ((32, nn.Sigmoid()), (4, nn.Sigmoid()), 2))
# model = NNet(7, ((16, nn.Sigmoid()), (4, nn.Sigmoid()), 2))
model = NNet(7, ((14, nn.Sigmoid()), (4, nn.Sigmoid()), 2))
# mostly big enough - very occasionally 99%
# model = NNet(7, ((13, nn.Sigmoid()), (4, nn.Sigmoid()), 2))

# Train
num_samples = 10000
train_data = generate_parity_data(num_samples)
print("train data hash: ", signature(train_data))
epochs = 50
learning_rate = 0.05
model.train(train_data, epochs, learning_rate=learning_rate, criterion=nn.MSELoss())

# inference accuracy
accuracy, correct, total = test_inference(model, num_samples=128)
print(f"Inference Accuracy: {accuracy:.2f}% ({correct} correct out of {total} samples)")
