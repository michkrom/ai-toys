#!/usr/bin/env python3
"""
parity.py — ML experiment: learn the parity bit of a 7-bit byte (0..127).

The network sees 7 input bits and predicts the even/odd parity bit
(popcount % 2) as a 2-way classification. Data is synthesized on the fly,
the net trains on random bytes, then is evaluated on ALL 128 possible bytes.

Like a lookup table this is trivially memorizable, but it is the classic
"first real ML" playground: a tiny exact function, clear labels, finite
test set -- the baseline pattern used by the other experiments.

Run directly (./parity.py) or as a cli.py subcommand:
    ./cli.py parity-learn [--acceleration auto|cpu|gpu|gpu:N] [--seed N] [--epochs N]
"""
import argparse
import random

import common  # sets warnings filter + shared torch init; import before torch
import torch
from torch import nn

from utils import NNet

DEFAULT_EPOCHS = 50
DEFAULT_SAMPLES = 10000
CONVERGE_AT = 1e-4   # stop training once mean loss is (displayively) 0
DEFAULT_SEED = 31415926535897932  # empirically trains to 100% (drives data RNG)


def next_byte_rand():
    return random.randint(0, 127)


byte = 0


def next_byte_lin():
    global byte
    byte = byte + 1 if byte < 127 else 0
    return byte


def bit_vec(byte, bits=7):
    return [(byte >> i) & 1 for i in range(bits - 1, -1, -1)]


def generate_parity_data(num_samples, next_byte=next_byte_rand):
    """tensor[7] bits --> tensor[2] (parity probability 0: [0], 1: [1])"""
    data = []
    for _ in range(num_samples):
        data_bits = bit_vec(next_byte())
        data_tensor = torch.tensor(data_bits, dtype=torch.float32)
        parity = sum(data_bits) % 2
        parity_probs = torch.tensor([0.5, 0.5], dtype=torch.float32)
        parity_probs[parity] = 1.0  # one-hot-ish target
        data.append((data_tensor, parity_probs))
    return data


def test_inference(model, device, num_samples=128):
    correct = 0
    model.eval()
    with torch.no_grad():
        for _ in range(num_samples):
            data_bits = bit_vec(next_byte_lin())
            data_tensor = torch.tensor(data_bits, dtype=torch.float32).to(device)
            actual_parity = sum(data_bits) % 2
            predicted_parity = int(model(data_tensor).argmax(dim=0).item())
            correct += predicted_parity == actual_parity
    return correct / num_samples * 100, correct, num_samples


def signature(data):
    import hashlib

    digest = hashlib.md5()
    for bits, par in data:
        v = 0
        for bit in bits:
            v = v * 2 + int(bit)
        v += int(par.argmax(dim=0).item()) * 128
        digest.update(bytes([v]))
    return digest.hexdigest()


# ------------------------------------------------------------------ experiment
def add_args(parser):
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)


def run(args):
    device = args.device
    if args.seed is None:  # keep the empirically-good default data seed
        random.seed(DEFAULT_SEED)
    # torch weight init left unseeded by default (as before); --seed seeds it

    print("== parity: learn the parity bit (popcount % 2) of a 7-bit byte ==")
    print("   input : 7 bits (0..127), e.g. 1011001 -> popcount 4 -> parity 0")
    print("   net   : 7 -> 14 -> 4 -> 2 logits, trained with MSE on one-hot labels")
    print("   test  : all 128 possible bytes (exact function, finite table)")
    print(f"   device: {common.describe_device(device)}\n")

    # active model: mostly reliable 100% (see history of commented archs below);
    # --hidden N overrides the hand-tuned 14 -> max(2, N//4) layering
    if args.hidden is None:
        model = NNet(7, ((14, nn.Sigmoid()), (4, nn.Sigmoid()), 2)).to(device)
    else:
        H = args.hidden
        model = NNet(7, ((H, nn.Sigmoid()), (max(2, H // 4), nn.Sigmoid()), 2)).to(device)
    print(f"   size   : {common.model_summary(model)[1]:,} parameters")

    train_data = generate_parity_data(args.samples)
    print("train data hash: ", signature(train_data))
    converged = model.train(
        train_data, args.epochs, learning_rate=0.05,
        criterion=nn.MSELoss(), device=device, converge_at=CONVERGE_AT,
    )
    if converged:
        print(f"  [converged at epoch {converged}, stopping early]")

    accuracy, correct, total = test_inference(model, device)
    print(f"Inference Accuracy: {accuracy:.2f}% ({correct} correct out of {total} samples)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common.add_torch_args(parser)
    add_args(parser)
    run(common.finish_args(parser.parse_args(argv)))


if __name__ == "__main__":
    main()