#!/usr/bin/env python3
"""
morseDecode.py — ML: decode morse symbol sequences into ASCII characters.

Same recipe as parity.py: synthesize data -> train a small net -> evaluate on
the whole table.

    input  : variable-length '.'/'-' sequence, e.g. ".-" -> 'A'
    output : 43-class softmax over the symbol table (A-Z, 0-9, punctuation)

Two models are compared:
  1. MLP baseline (parity-style): symbol sequence padded to 6 slots (one-hot
     per slot: pad/dot/dash) -> flat 18 dims -> MLP -> 43 logits.
  2. GRU: the same padded sequence fed as (6,3) time steps; the recurrent net
     is the architecturally honest way to read a variable-length symbol code.

NOTE: with a clean, finite table this is still a 43-row lookup-style task, but
unlike morse.py (char -> code, fixed input) the *input* here is a true
variable-length sequence, so the GRU/MLP choice actually matters once inputs
get noisy or hand-keyed. The hard/unsolved variant is decoding a continuous
stream with no inter-symbol gaps (segmentation + language model).
"""
import random
import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import morseCode
from utils import DatasetFeeder, NNet

MAX_LEN = max(len(code) for _, code in morseCode.morseCode)  # 6 slots
N_CLASSES = len(morseCode.morseCode)                         # 43

# sanity: every char must have a unique code (it does) and fit in MAX_LEN slots
codes = [code for _, code in morseCode.morseCode]
assert len(set(codes)) == N_CLASSES, "duplicate morse codes in table"
assert MAX_LEN <= 6


# ------------------------------------------------------------------ data
def code_to_input(code):
    """'.-' -> tensor(6,3): one-hot over (pad, dot, dash) in each slot."""
    t = torch.zeros(MAX_LEN, 3)
    t[:, 0] = 1.0
    for i, c in enumerate(code):
        if c == ".":
            t[i] = torch.tensor([0.0, 1.0, 0.0])
        elif c == "-":
            t[i] = torch.tensor([0.0, 0.0, 1.0])
    return t


def gen_data(num_samples):
    """random table entry -> (padded symbol one-hot, class id)"""
    data = []
    for _ in range(num_samples):
        idx = random.randrange(N_CLASSES)
        data.append(
            (code_to_input(morseCode.morseCode[idx][1]),
             torch.tensor(idx, dtype=torch.long))
        )
    return data


# ------------------------------------------------------------------ training
def train_model(model, data, epochs=30, lr=1e-3, batch_size=32):
    loader = DataLoader(DatasetFeeder(data), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    nn.Module.train(model)  # toggles both the wrapper and its NNet child
    for epoch in range(epochs):
        total = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        print(f"  epoch {epoch + 1:3d}  loss {total / len(data):.4f}")


def evaluate(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for idx, (char, code) in enumerate(morseCode.morseCode):
            logits = model(code_to_input(code).unsqueeze(0))
            pred = int(logits.argmax(1))
            if pred == idx:
                correct += 1
            else:
                print(f"  {char} |{code}| -> {morseCode.morseCode[pred][0]}")
    print(f"  {int(100 * correct / N_CLASSES)}% ({correct}/{N_CLASSES})")


def morse_to_text(model, morse_text):
    """space-separated morse -> plain text, decoded by the trained net."""
    model.eval()
    out = []
    with torch.no_grad():
        for token in morse_text.split():
            logits = model(code_to_input(token).unsqueeze(0))
            out.append(morseCode.morseCode[int(logits.argmax(1))][0])
    return "".join(out)


# ------------------------------------------------------------------ models
class FlattenMLP(nn.Module):
    """adapter: (B,6,3) sequence input -> flat 18 dims for the MLP baseline"""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x.reshape(x.size(0), -1))


class GRUClassifier(nn.Module):
    """reads the (6,3) symbol sequence and classifies the character."""

    def __init__(self, input_dim=3, hidden=32, out=N_CLASSES):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, out)

    def forward(self, x):
        out, _ = self.gru(x)          # (B,6,hidden)
        return self.fc(out[:, -1])    # classify after last (padded) step


random.seed(0)
torch.manual_seed(0)

print("== 1) parity-style MLP on padded fixed-length input ==")
mlp = FlattenMLP(NNet(MAX_LEN * 3, ((64, nn.ReLU()), (64, nn.ReLU()), N_CLASSES)))
train_model(mlp, gen_data(10000))
evaluate(mlp)

print("\n== 2) GRU sequence model on the same data ==")
gru = GRUClassifier()
train_model(gru, gen_data(10000))
evaluate(gru)

print("\n== decode demo (GRU) ==")
print("  SOS  :", morse_to_text(gru, "... --- ..."))
print("  HELLO:", morse_to_text(gru, ".... . .-.. .-.. ---"))