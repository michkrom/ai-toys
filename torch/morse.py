import warnings

warnings.filterwarnings("ignore")
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import DatasetFeeder, NNet
import morseCode
import random


def morse_to_onehot_list(mcode):
    """mcode string of .- to a vector of probabilities where each 2 are p(dot) p(dash) probabilities"""
    #   0 0 none
    # . 1 0 dot
    # - 0 1 dash
    v = []
    for m in mcode:
        v = v + ([1, 0.5, 0.5] if m == "." else [0.5, 1, 0.5])
    while len(v) < 3 * 6:
        v = v + [0.5, 0.5, 1]
    return v


def gen_data(dataLen):
    """generate x: char -> [ 6 x p(dot) p(dash)  p(silence) ]"""
    data = []
    alphabetLen = len(morseCode.morseCode)
    for i in range(0, dataLen):
        x = random.randint(0, alphabetLen)
        code = morseCode.morseCode[x][1] if x < alphabetLen else ''
        y = morse_to_onehot_list(code)
        x_tensor = torch.tensor([x], dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        data.append((x_tensor, y_tensor))
    return data


def probs_to_morse(y):
    yv = y.view(6, 3)
    m = ""
    for i in torch.argmax(yv, dim=1):
        m = m + [".", "-", " "][i.item()]
    return m


def check_data(data):
    for x, y in data:
        m1 = morseCode.morseCode[int(x[0])][1] if x < len(morseCode.morseCode) else ''
        m2 = probs_to_morse(y)
        if m1 != m2.strip():
            print(x, m1, m2)
            return


def test_inference():
    good = 0
    for i in range(0, len(morseCode.morseCode)):
        y = model(torch.tensor([i], dtype=torch.float32))
        m = morseCode.morseCode[i]
        mi = probs_to_morse(y).strip()
        if m[1] == mi:
            good = good+1
        else:
            print(f"{m[0]} |{m[1]}|, |{mi}| ")
    print(f"{int(100*good/len(morseCode.morseCode))}%")


# deeper netowrk performed much better, with 2 layers best accy was 75%, this 4 layers gets to 100%
model = NNet(1, ((64, nn.Sigmoid()), (64, nn.Sigmoid()), (44, nn.Sigmoid()), (6 * 3)))
print(model)

train_data = gen_data(10000)
check_data(train_data)

# gross
model.train(
    train_data, epochs=30, learning_rate=0.01, criterion=nn.MSELoss()
)
test_inference()

# tuneup
train_data = gen_data(10000)
model.train(
    train_data, epochs=30, learning_rate=0.001, criterion=nn.MSELoss()
)
test_inference()

train_data = gen_data(10000)
model.train(
    train_data, epochs=50, learning_rate=0.0001, criterion=nn.MSELoss()
)
test_inference()

