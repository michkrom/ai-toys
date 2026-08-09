# torch — PyTorch playground

Small ML experiments in PyTorch: each toy synthesizes its own data, trains a
tiny net, and evaluates on the complete finite table. Everything runs through
one CLI with a single shared torch init (device / seed / threads).

## Requirements

```bash
pip install torch        # numpy comes with most torch installs
# CUDA GPUs are used automatically when present (--device auto)
```

## Usage — one CLI for everything

```bash
cd torch

./cli.py morse-encode    [--epochs N] [--samples N] [--acceleration auto|cpu|gpu|gpu:N] [--seed N] [--threads N]
./cli.py morse-decode    [same flags] [--model mlp|gru]
./cli.py parity-learn    [same flags]
./cli.py morse-translate [--verbose] [FILE]      # non-ML reference translator
```

Common flags (from `common.py`, applied to all ML subcommands):

| flag | meaning |
|---|---|
| `--acceleration auto\|cpu\|gpu\|gpu:N` (alias `--accel`) | compute acceleration; `auto` (default) uses **GPU** (cuda:0) when available, else CPU; `cpu` forces CPU; `gpu:N` picks GPU number N (`nvidia-smi` to list); `cuda:N` accepted too |
| `--seed N` | seed python+torch RNGs (default: per-experiment defaults) |
| `--threads N` | limit torch CPU threads (useful on shared/loaded machines) |
| `--epochs N`, `--samples N` | training length / dataset size |

Every run prints an explicit banner naming the device actually used, e.g.
`[cli] run=decode device=cuda:0 (NVIDIA RTX A2000 12GB)`.

Examples:

```bash
./cli.py morse-decode --model gru             # defaults to GPU (cuda:0)
./cli.py morse-encode --accel gpu:1           # pick the other GPU (cuda:1)
./cli.py parity-learn --acceleration cpu      # no GPU
./cli.py parity-learn --epochs 100 --threads 4
./cli.py morse-decode --model mlp --seed 7    # reproducible run
./cli.py morse-translate <<< "SOS"           # ... --- ...
```

The experiment scripts can also still be run directly (`./morse.py`, etc.),
but the CLI is the intended entry point.

## The ML experiments

### `morse-encode` (`morse.py`) — learn char → morse code

One-hot character index (43 classes) → 6 symbol slots, each classified as
dot / dash / silence (padding); per-position cross-entropy. Effectively a
lookup-table task — an NN is convenient here, not magical; the problems
where sequences/audio actually need learning are the `decode` experiment.

### `morse-decode` (`morseDecode.py`) — learn morse code → char

The reverse problem: variable-length `'.'`/`'-'` sequence → 43-class
character. Three models, `--model <mlp|gru|rnn|stream>`:

1. `mlp` — parity-style fixed-length MLP on the padded 18-dim vector (baseline)
2. `gru` — padded (6,3) sequence read as time steps
3. `rnn` — **true symbol-by-symbol RNN**: only the real `.`/`-` tokens are
   fed, one per timestep (`pack_padded_sequence`, zero padding steps); the
   final hidden state decides the char, so the net must remember the whole
   prefix — morse codes share prefixes (`.`=E, `..`=I, `...`=S), and the run
   ends with a prefix-trace demo showing the net change its mind as symbols
   arrive
4. `stream` — **Option A streaming decode**: the same RNN, but letters are
   decoded one after another from a live stream; the *client* detects each
   letter break (from gap timing), classifies from the accumulated hidden
   state, then **resets the RNN state to zeros** for the next letter — no
   gap tokens in the input. Shows per-symbol confidence: the correct letter's
   probability jumps only once its last symbol is in memory

Every run prints the model's **topology and parameter count** (e.g.
`topology: Embedding 2->8, GRU 8->32 x1, Linear 32->43 (5,467 parameters)`).
All models reach 100% (43/43) and end with a live demo decoding
space-separated morse → text (SOS, HELLO). The still-unsolved variant —
segmenting a continuous stream with no inter-symbol gaps — needs a search /
language model and is discussed in the module docstring.

### `parity-learn` (`parity.py`) — learn the parity bit

7 input bits → parity bit (popcount % 2), 2-way classification, trained with
MSE on one-hot labels; evaluated on all 128 bytes. The classic "first real
ML" playground and the baseline pattern for the other experiments.

## Support files

- `cli.py` — single CLI entry point (subcommands, one torch init per run).
- `common.py` — shared torch init: warnings, `--device/--seed/--threads`.
- `utils.py` — `NNet` (generic MLP with train loop) and `DatasetFeeder`.
- `morse_data.py` — morse table (43 entries) + pure text↔code helpers, used by
  the experiments and the `translate` command (no torch, import-safe).
  (The old `morseCode.py` CLI was folded into the `morse-translate` subcommand.)