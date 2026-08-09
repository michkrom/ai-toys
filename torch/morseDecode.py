#!/usr/bin/env python3
"""
morseDecode.py — ML experiment: decode morse symbol sequences into ASCII.

    input  : variable-length '.'/'-' sequence, e.g. ".-" -> 'A'
    output : 43-class softmax over the symbol table (A-Z, 0-9, punctuation)

Three models, selected with --model:
  mlp: parity-style MLP on padded fixed-length input (default)
  gru: recurrent net reading the (6,3) symbol sequence -- the architecturally
       honest way to handle variable-length input
  rnn: true symbol-by-symbol RNN -- only the real '.'/'-' tokens are fed, one
       per timestep (packed, no padding steps); the final hidden state is the
       memory that classifies the char, so it must remember the whole prefix
  stream: Option A streaming decode -- same RNN, but letters arrive one after
       another and the CLIENT (from gap timing) detects each letter break,
       classifies from the accumulated hidden state, then resets it to zeros
       for the next letter. No gap tokens in the input; the confidence in the
       correct letter only materializes after its last symbol arrives.

Every run prints the model's topology and parameter count.

Same recipe as parity.py: synthesize data -> train a small net -> evaluate on
the whole table. The still-unsolved variant (dealing with a continuous stream
that has no inter-symbol gaps) would need segmentation + a language model.

Run directly (./morseDecode.py) or as a cli.py subcommand:
    ./cli.py morse-decode [--model mlp|gru] [--acceleration auto|cpu|gpu|gpu:N] [--seed N]
                    [--epochs N]
"""
import argparse
import random

import common  # sets warnings filter + shared torch init; import before torch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import morse_data
from utils import DatasetFeeder, NNet

MAX_LEN = max(len(code) for _, code in morse_data.MORSE_CODES)  # 6 slots
N_CLASSES = len(morse_data.MORSE_CODES)                         # 43
DEFAULT_EPOCHS = 30
DEFAULT_SAMPLES = 10000
CONVERGE_AT = 1e-4   # stop training once mean loss is (displayively) 0

# stream detection is done by the CLIENT (timing), not the model: the RNN
# only ever sees '.'/'-' symbol ids (see code_to_seq).

# sanity: unique codes, all fitting in MAX_LEN slots
codes = [code for _, code in morse_data.MORSE_CODES]
assert len(set(codes)) == N_CLASSES, "duplicate morse codes in table"
assert MAX_LEN <= 6


# ------------------------------------------------------------------ data
def code_to_input(code, device=None):
    """'.-' -> tensor(6,3): one-hot over (pad, dot, dash) in each slot."""
    t = torch.zeros(MAX_LEN, 3)
    t[:, 0] = 1.0
    for i, c in enumerate(code):
        if c == ".":
            t[i] = torch.tensor([0.0, 1.0, 0.0])
        elif c == "-":
            t[i] = torch.tensor([0.0, 0.0, 1.0])
    return t.to(device) if device is not None else t


def gen_data(num_samples, device=None):
    """random table entry -> (padded symbol one-hot, class id)"""
    data = []
    for _ in range(num_samples):
        idx = random.randrange(N_CLASSES)
        x = code_to_input(morse_data.MORSE_CODES[idx][1], device)
        data.append((x, torch.tensor(idx, dtype=torch.long, device=device)))
    return data


# ------------------------------------------------------------------ training
def train_model(model, data, epochs=DEFAULT_EPOCHS, lr=1e-3, batch_size=32):
    loader = DataLoader(DatasetFeeder(data), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        mean_loss = total / len(data)
        print(f"  epoch {epoch + 1:3d}  loss {mean_loss:.6f}")
        if mean_loss <= CONVERGE_AT:
            print(f"  [converged at epoch {epoch + 1}, stopping early]")
            break


def evaluate(model, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for idx, (char, code) in enumerate(morse_data.MORSE_CODES):
            logits = model(code_to_input(code, device).unsqueeze(0))
            pred = int(logits.argmax(1))
            if pred == idx:
                correct += 1
            else:
                print(f"  {char} |{code}| -> {morse_data.MORSE_CODES[pred][0]}")
    print(f"  {int(100 * correct / N_CLASSES)}% ({correct}/{N_CLASSES})")


def morse_to_text(model, morse_text, device):
    """space-separated morse -> plain text, decoded by the trained net."""
    model.eval()
    out = []
    with torch.no_grad():
        for token in morse_text.split():
            logits = model(code_to_input(token, device).unsqueeze(0))
            out.append(morse_data.MORSE_CODES[int(logits.argmax(1))][0])
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


class SymbolRNN(nn.Module):
    """true variable-length decoder: one symbol per timestep, no padding.

    Reads '.'/'-' symbol ids one at a time (pack_padded_sequence); the final
    hidden state after the LAST real symbol is the memory used for the
    classification -- the net must remember the whole prefix because morse
    codes share prefixes ('.'=E vs '.-'=A vs '.--'=W ...).
    """

    def __init__(self, vocab=2, embed=8, hidden=32, out=N_CLASSES):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.gru = nn.GRU(embed, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, out)

    def forward(self, seqs, lengths):
        embedded = self.emb(seqs)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.fc(hidden[-1])  # memory after the last real symbol


# ------------------------------------------------------------------ rnn data
def code_to_seq(code):
    """'.-' -> tensor of symbol ids (0=dot, 1=dash), true variable length."""
    return torch.tensor([0 if c == "." else 1 for c in code], dtype=torch.long)


def gen_seq_data(num_samples):
    data = []
    for _ in range(num_samples):
        idx = random.randrange(N_CLASSES)
        code = morse_data.MORSE_CODES[idx][1]
        data.append((code_to_seq(code), torch.tensor(idx, dtype=torch.long)))
    return data


def collate_seqs(batch):
    """pad codes to the longest in the batch; real lengths shipped separately."""
    seqs, ys = zip(*batch)
    max_len = max(len(s) for s in seqs)
    padded = torch.stack([F.pad(s, (0, max_len - len(s))) for s in seqs])
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    return padded, torch.stack(ys), lengths


def train_seq_model(model, data, device, epochs=DEFAULT_EPOCHS, lr=1e-3, batch_size=32):
    loader = DataLoader(DatasetFeeder(data), batch_size=batch_size, shuffle=True,
                        collate_fn=collate_seqs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for seqs, ys, lengths in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(seqs.to(device), lengths), ys.to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(ys)
        mean_loss = total / len(data)
        print(f"  epoch {epoch + 1:3d}  loss {mean_loss:.6f}")
        if mean_loss <= CONVERGE_AT:
            print(f"  [converged at epoch {epoch + 1}, stopping early]")
            break


def eval_seq(model, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for idx, (char, code) in enumerate(morse_data.MORSE_CODES):
            seq = code_to_seq(code).unsqueeze(0).to(device)
            lengths = torch.tensor([len(code)])
            pred = int(model(seq, lengths).argmax(1))
            if pred == idx:
                correct += 1
            else:
                print(f"  {char} |{code}| -> {morse_data.MORSE_CODES[pred][0]}")
    print(f"  {int(100 * correct / N_CLASSES)}% ({correct}/{N_CLASSES})")


def morse_to_text_seq(model, morse_text, device):
    """space-separated morse -> plain text (variable-length RNN decoder)."""
    model.eval()
    out = []
    with torch.no_grad():
        for token in morse_text.split():
            seq = code_to_seq(token).unsqueeze(0).to(device)
            pred = int(model(seq, torch.tensor([len(token)])).argmax(1))
            out.append(morse_data.MORSE_CODES[pred][0])
    return "".join(out)


def prefix_traces(model, device):
    """all letters: which char the net predicts after each symbol prefix."""
    model.eval()
    with torch.no_grad():
        for idx, (char, code) in enumerate(morse_data.MORSE_CODES):
            steps, h = [], None
            for ch in code:
                x = model.emb(code_to_seq(ch).unsqueeze(0).to(device))
                _, h = model.gru(x, h)
                pred = morse_data.MORSE_CODES[int(model.fc(h[-1]).argmax(1))][0]
                steps.append(f"{ch} ==> {pred}")
            print(f"  {char} {code:<6} : {', '.join(steps)}")


def stream_morse_to_text(model, segments, device):
    """Option A: the client detects each letter break and resets the RNN.

    segments = ['...', '---', '...'] (letter breaks already located by the
    client, e.g. from the 3-unit gap timing). Symbols of one letter are fed
    one by one into a persistent hidden state; at the break the client
    classifies from the accumulated state, emits the letter, and resets the
    state to zeros for the next letter. There are NO gap tokens in the input.
    """
    model.eval()
    out, h = [], None
    with torch.no_grad():
        for seg in segments:
            for ch in seg:                     # feed this letter's symbols
                x = model.emb(code_to_seq(ch).unsqueeze(0).to(device))
                _, h = model.gru(x, h)         # h None -> zeros (start letter)
            pred = int(model.fc(h[-1]).argmax(1))
            out.append(morse_data.MORSE_CODES[pred][0])
            h = None                           # <-- reset on detected break
    return "".join(out)


def confidence_traces(model, device):
    """all letters: P(correct char) after each symbol (client commits at break)."""
    model.eval()
    with torch.no_grad():
        for idx, (char, code) in enumerate(morse_data.MORSE_CODES):
            h, probs = None, []
            for ch in code:
                x = model.emb(code_to_seq(ch).unsqueeze(0).to(device))
                _, h = model.gru(x, h)
                p = F.softmax(model.fc(h[-1]), dim=-1)[0, idx].item()
                probs.append(f"{p:.2f}")
            print(f"  {char} {code:<6} : P({char}) per symbol = {' '.join(probs)}")


# ------------------------------------------------------------------ experiment
def add_args(parser):
    parser.add_argument("--model", choices=("mlp", "gru", "rnn", "stream"),
                        default="mlp")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)



def run(args):
    device = args.device
    if args.seed is None:
        random.seed(0)
    torch.manual_seed(0)

    if args.model in ("rnn", "stream"):
        if args.model == "rnn":
            print("== symbol-RNN: one symbol per step, variable length (memory) ==")
        else:
            print("== stream: same RNN, client detects letter breaks and resets (Option A) ==")
        hidden = args.hidden or 8   # minimal-but-working default (floor is 8; 4 fails)
        model = SymbolRNN(vocab=2, embed=max(2, hidden // 4), hidden=hidden).to(device)
        topology, params = common.model_summary(model)
        print(f"   device: {common.describe_device(device)}")
        print(f"   topology: {topology}   ({params:,} parameters)")
        train_seq_model(model, gen_seq_data(args.samples), device, epochs=args.epochs)
        eval_seq(model, device)
        print("\n== decode demo (trained net) ==")
        if args.model == "rnn":
            print("  SOS  :", morse_to_text_seq(model, "... --- ...", device))
            print("  HELLO:", morse_to_text_seq(model, ".... . .-.. .-.. ---", device))
            print()
            print("  prefix memory: what the net predicts after each symbol (all 43 letters):")
            prefix_traces(model, device)
        else:
            # client already found the breaks; RNN just consumed each letter's
            # symbols and got reset by the client at every break
            print("  SOS  :", stream_morse_to_text(model, ["...", "---", "..."], device))
            print("  HELLO:", stream_morse_to_text(model, ["....", ".", ".-..", ".-..", "---"], device))
            print("\n  per-symbol confidence of each letter (client commits at the break):")
            confidence_traces(model, device)
    else:
        if args.model == "mlp":
            print("== parity-style MLP on padded fixed-length input ==")
            H = args.hidden or 8   # minimal-but-working default (floor is 4)
            model = FlattenMLP(
                NNet(MAX_LEN * 3, ((H, nn.ReLU()), (H, nn.ReLU()), N_CLASSES))
            ).to(device)
        else:
            print("== GRU sequence model ==")
            model = GRUClassifier(hidden=args.hidden or 8).to(device)  # floor 8; 4 fails
        topology, params = common.model_summary(model)
        print(f"   device: {common.describe_device(device)}")
        print(f"   topology: {topology}   ({params:,} parameters)")

        train_model(model, gen_data(args.samples, device), epochs=args.epochs)
        evaluate(model, device)

        print("\n== decode demo (trained net) ==")
        print("  SOS  :", morse_to_text(model, "... --- ...", device))
        print("  HELLO:", morse_to_text(model, ".... . .-.. .-.. ---", device))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common.add_torch_args(parser)
    add_args(parser)
    run(common.finish_args(parser.parse_args(argv)))


if __name__ == "__main__":
    main()