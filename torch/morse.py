#!/usr/bin/env python3
"""
morse.py — ML experiment: learn to spell a character in morse code.

    input  : one-hot vector of the character index (43 classes: A-Z, 0-9, ..)
    output : 6 symbol slots, each classified as dot / dash / silence
             (padding) -> 6*3 logits
    loss   : per-position CrossEntropyLoss

Because the input is just a character index, this is effectively a
lookup-table task. The interesting generalization variants (audio-in,
noisy sequence decoding) are the subject of morseDecode.py.

Run directly (./morse.py) or as a cli.py subcommand:
    ./cli.py morse-encode [--acceleration auto|cpu|gpu|gpu:N] [--seed N] [--epochs N]
"""
import argparse
import random

import common  # sets warnings filter + shared torch init; import before torch
import torch
from torch import nn
from torch.nn import functional as F

import morse_data
from utils import NNet

MAX_SYMBOLS = 6
N_CLASSES = len(morse_data.MORSE_CODES)              # 43
SYMBOL_TARGETS = (".", "-", " ")
DEFAULT_EPOCHS = 30
DEFAULT_SAMPLES = 20000
CONVERGE_AT = 1e-4   # stop training once mean loss is (displayively) 0


# ------------------------------------------------------------------ data
def char_to_input(index):
    """one-hot vector for a character index -- a proper NN input."""
    x = torch.zeros(N_CLASSES)
    x[index] = 1.0
    return x


def code_to_target(mcode):
    """'.-' -> tensor(6,3) one-hot over (dot, dash, silence)."""
    target = torch.zeros(MAX_SYMBOLS, 3)
    target[:, 2] = 1.0  # silence by default (padding)
    for pos, m in enumerate(mcode):
        target[pos] = torch.tensor([1.0, 0.0, 0.0] if m == "." else [0.0, 1.0, 0.0])
    return target


def gen_data(num_samples):
    data = []
    for _ in range(num_samples):
        index = random.randrange(N_CLASSES)
        data.append((char_to_input(index), code_to_target(morse_data.MORSE_CODES[index][1])))
    return data


# ------------------------------------------------------------------ decode / eval
def logits_to_morse(logits):
    """tensor(18) logits -> string of '.', '-' and ' '."""
    return "".join(SYMBOL_TARGETS[i] for i in logits.view(6, 3).argmax(dim=1))


def seq_cross_entropy(logits, targets):
    """(B,18) logits + (B,6,3) one-hot targets -> CE over all 6 positions."""
    return F.cross_entropy(logits.view(-1, 3), targets.view(-1, 3).argmax(dim=-1))


# ------------------------------------------------------------------ experiment
def add_args(parser):
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)


def run(args):
    device = args.device
    H = args.hidden or 64
    if args.seed is None:  # common.init_torch already seeded if --seed given
        random.seed(0)
        torch.manual_seed(0)

    model = NNet(N_CLASSES, ((H, nn.ReLU()), (H, nn.ReLU()), MAX_SYMBOLS * 3)).to(device)
    print(f"== encode: char -> morse code, device: {common.describe_device(device)}")
    print(model)
    print(f"   size: {common.model_summary(model)[1]:,} parameters")

    train_data = gen_data(args.samples)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.stack([x for x, _ in train_data]),
            torch.stack([y for _, y in train_data]),
        ),
        batch_size=32, shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    model.training = True  # NNet shadows .train(); no dropout here so the flag suffices

    for epoch in range(args.epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = seq_cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        mean_loss = total / len(train_data)
        print(f"Epoch: {epoch + 1:3d}, Loss: {mean_loss:.6f}")
        if mean_loss <= CONVERGE_AT:
            print(f"  [converged at epoch {epoch + 1}, stopping early]")
            break

    # evaluate on all 43 table entries
    model.eval()
    good = 0
    with torch.no_grad():
        for index in range(N_CLASSES):
            expected = morse_data.MORSE_CODES[index][1]
            predicted = logits_to_morse(model(char_to_input(index).to(device))).replace(" ", "")
            good += expected == predicted
            if expected != predicted:
                print(f"  {morse_data.MORSE_CODES[index][0]} |{expected}|, |{predicted}|")
    print(f"{int(100 * good / N_CLASSES)}% ({good}/{N_CLASSES})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common.add_torch_args(parser)
    add_args(parser)
    run(common.finish_args(parser.parse_args(argv)))


if __name__ == "__main__":
    main()