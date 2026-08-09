# torch — PyTorch playground

Small ML experiments in PyTorch. Each script is self-contained: a printed
brief, synthesized training data, a small net, and an evaluation over the
complete finite table.

## Requirements

```bash
pip install torch        # numpy comes with most torch installs
pip install matplotlib   # optional, only for utils.NNet.visualize_training()
```

## The ML scripts

### `morse.py` — NN that learns morse encoding

Trains a small MLP to spell a **character** in morse code: one-hot vector of
the character index (43 symbols: A–Z, 0–9, punctuation) → 6 symbol slots,
each classified as dot / dash / silence (padding); per-position
cross-entropy.

```bash
./morse.py        # or: python3 morse.py
```

Output: loss per epoch, then evaluation over all 43 symbols.

```
Epoch: 30, Loss: 0.0000
100% (43/43)
```

Note: because the input is a character index, this is effectively a
lookup-table task — an NN is convenient here, not magical. The problems
where sequences/audio actually need learning are covered by `morseDecode.py`
and its docstring.

### `parity.py` — NN that learns the parity bit

Trains a small MLP to predict the parity bit (popcount % 2) of a 7-bit byte
(128 possible inputs), then evaluates on all of them.

```bash
./parity.py        # or: python3 parity.py
```

Output: a short brief (input/net/test), loss per epoch, then
`Inference Accuracy: 100.00% (128 correct out of 128 samples)`.

### `morseDecode.py` — NN that decodes morse → ASCII

The reverse problem: a variable-length `'.'`/`'-'` sequence (e.g. `.- → A`)
maps to the 43-class character. Same recipe as `parity.py` (synth data →
train small net → evaluate the whole table), comparing two models:

1. parity-style MLP on padded fixed-length input
2. GRU sequence model (the architecturally honest choice for variable-length input)

```bash
./morseDecode.py
```

Both reach 100% (43/43); a live demo at the end decodes space-separated
morse → text (SOS, HELLO). The still-unsolved variant — segmenting a
continuous stream with no inter-symbol gaps — is discussed in the docstring.

## Support files

- `utils.py` — shared plumbing: `NNet` (generic MLP with train loop,
  `visualize_training()`) and `DatasetFeeder`.
- `morseCode.py` — plain (non-ML) text→morse translator; also the data table
  that `morse.py` and `morseDecode.py` import for training data.

### `morseCode.py` — plain reference translator (no ML)

```bash
echo "SOS" | ./morseCode.py            # → ... --- ...
./morseCode.py message.txt             # translate a file
./morseCode.py --verbose <<< "HI"       # char-by-char breakdown
./morseCode.py --help                  # full usage
```