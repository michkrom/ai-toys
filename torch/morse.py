#!/usr/bin/env python3
"""
Morse encoder toy: learn to spell a character (A-Z, 0-9, punctuation,
43 classes) in morse code.

    input  : one-hot vector of the character index (43 dims)
    output : 6 symbol slots, each classified as dot / dash / silence
             (padding) -> 6*3 logits
    loss   : per-position CrossEntropyLoss

Because the input is just a character index, this is effectively a
lookup-table task. The interesting generalization variants (audio-in,
noisy sequence decoding) are the subject of morseDecode.py.
"""
import random
import warnings

warnings.filterwarnings("ignore")

import torch
from torch.nn import functional as F

import morseCode
from utils import NNet

MAX_SYMBOLS = 6
N_SYMBOLS = len(morseCode.morseCode)  # 43 rows: letters, digits, punctuation
SYMBOL_TARGETS = (".", "-", " ")


# ---------------------------------------------------------------- data
def char_to_input(index):
    """one-hot vector for a character index -- a proper NN input."""
    x = torch.zeros(N_SYMBOLS)
    x[index] = 1.0
    return x


def code_to_target(mcode):
    """'.-' -> tensor(6,3) one-hot over (dot, dash, silence)."""
    target = torch.zeros(MAX_SYMBOLS, 3)
    target[:, 2] = 1.0  # silence by default (padding)
    for pos, m in enumerate(mcode):
        target[pos] = torch.tensor([1.0, 0.0, 0.0] if m == "." else [0.0, 1.0, 0.0])
    return target


def gen_data(data_len):
    data = []
    for _ in range(data_len):
        index = random.randrange(N_SYMBOLS)  # [0, N) -- no off-by-one
        code = morseCode.morseCode[index][1]
        data.append((char_to_input(index), code_to_target(code)))
    return data


# ---------------------------------------------------------------- decode / eval
def logits_to_morse(logits):
    """tensor(18) logits -> string of '.', '-' and ' '."""
    return "".join(SYMBOL_TARGETS[i] for i in logits.view(6, 3).argmax(dim=1))


def check_data(data):
    """Verify the encoders are consistent: index -> code -> target -> code."""
    for x, y in data:
        index = int(x.argmax().item())
        code = morseCode.morseCode[index][1]
        decoded = logits_to_morse(y.view(-1)).replace(" ", "")
        assert code == decoded, (index, code, decoded)


def test_inference():
    good = 0
    for index in range(N_SYMBOLS):
        char, expected = morseCode.morseCode[index]
        predicted = logits_to_morse(model(char_to_input(index))).replace(" ", "")
        good += expected == predicted
        if expected != predicted:
            print(f"{char} |{expected}|, |{predicted}|")
    print(f"{int(100 * good / N_SYMBOLS)}% ({good}/{N_SYMBOLS})")


def seq_cross_entropy(logits, targets):
    """(B,18) logits + (B,6,3) one-hot targets -> CE over all 6 positions."""
    return F.cross_entropy(logits.view(-1, 3), targets.view(-1, 3).argmax(dim=-1))


# ---------------------------------------------------------------- model
torch.manual_seed(0)
random.seed(0)

# small net: one-hot(43) -> 64 -> 64 -> 18 logits
model = NNet(N_SYMBOLS, ((64, torch.nn.ReLU()), (64, torch.nn.ReLU()), MAX_SYMBOLS * 3))
print(model)

train_data = gen_data(20000)
check_data(train_data)

model.train(train_data, epochs=30, learning_rate=0.001, criterion=seq_cross_entropy)
test_inference()