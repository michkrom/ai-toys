# torch — PyTorch playground

Small ML experiments in PyTorch: each toy synthesizes its own data, trains a
tiny net, evaluates on the complete finite table, and stops early once the
loss reaches ~0 (1e-4). Everything runs through one CLI with a single shared
torch init (acceleration / seed / threads).

## Requirements

```bash
pip install torch        # numpy comes with most torch installs
# CUDA GPUs are used automatically (--acceleration auto)
```

## Usage — one CLI for everything

Subcommands are named `<domain>-<action>` so the set can grow:

```bash
cd torch

./cli.py morse-encode    [--epochs N] [--samples N] [--acceleration auto|cpu|gpu|gpu:N] [--seed N] [--threads N]
./cli.py morse-decode    [same flags] [--model mlp|gru|rnn|stream]
./cli.py parity-learn    [same flags]
./cli.py morse-translate [--verbose] [FILE]      # non-ML reference translator
```

Common flags (from `common.py`, applied to all ML subcommands):

| flag | meaning |
|---|---|
| `--acceleration auto\|cpu\|gpu\|gpu:N` (alias `--accel`) | compute acceleration; `auto` (default) uses **GPU** (cuda:0) when available, else CPU; `cpu` forces CPU; `gpu:N` picks GPU number N (`nvidia-smi` to list); `cuda:N` accepted too |
| `--seed N` | seed python+torch RNGs (default: per-experiment defaults) |
| `--threads N` | limit torch CPU threads (useful on shared/loaded machines) |
| `--epochs N`, `--samples N` | training length / dataset size (`--epochs` is a *cap*: training stops at loss ≤ 1e-4) |

Every run prints an explicit banner naming the device actually used, e.g.
`[cli] run=morse-decode device=cuda:0 (NVIDIA RTX A2000 12GB)`.

Examples:

```bash
./cli.py morse-decode --model gru             # defaults to GPU (cuda:0)
./cli.py morse-encode --accel gpu:1           # pick the other GPU (cuda:1)
./cli.py parity-learn --acceleration cpu      # no GPU
./cli.py parity-learn --epochs 100 --threads 4
./cli.py morse-decode --model rnn --seed 7    # reproducible run
./cli.py morse-translate <<< "SOS"           # ... --- ...
```

The experiment scripts can also still be run directly (`./morse.py`, etc.),
but the CLI is the intended entry point.

## The ML experiments

### `morse-encode` (`morse.py`) — learn char → morse code

One-hot character index (43 classes) → 6 symbol slots, each classified as
dot / dash / silence (padding); per-position cross-entropy. Effectively a
lookup-table task — an NN is convenient here, not magical; the problems
where sequences/audio actually need learning are the `morse-decode`
experiment.

### `morse-decode` (`morseDecode.py`) — learn morse code → char

The reverse problem: variable-length `'.'`/`'-'` sequence → 43-class
character. Four models, `--model <mlp|gru|rnn|stream>`:

1. `mlp` — parity-style fixed-length MLP on the padded 18-dim vector (baseline)
2. `gru` — padded (6,3) sequence read as time steps
3. `rnn` — **true symbol-by-symbol RNN**: only the real `.`/`-` tokens are
   fed, one per timestep (`pack_padded_sequence`, zero padding steps); the
   final hidden state decides the char, so the net must remember the whole
   prefix — morse codes share prefixes (`.`=E, `..`=I, `...`=S), and the run
   ends with a per-symbol trace of all 43 letters showing the net change its
   mind as symbols arrive
4. `stream` — **Option A streaming decode**: the same RNN, but letters are
   decoded one after another from a live stream; the *client* detects each
   letter break (from gap timing), classifies from the accumulated hidden
   state, then **resets the RNN state to zeros** for the next letter — no
   gap tokens in the input. Prints per-symbol confidence for all 43 letters:
   the correct letter's probability jumps only once its last symbol is in
   memory (e.g. `J .--- : 0.00 0.00 0.00 1.00`)

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

## Model sizes vs. problem size

These toys are finite-table tasks, so the interesting question is how much
capacity they actually *need*. Measured (morse-decode RNN with `embed=8`):

| model | params | × 43-entry table | accuracy |
|---|---|---|---|
| decode `mlp` | 8,171 | 190× | 100% |
| decode `gru` | 4,971 | 116× | 100% |
| decode `rnn` hidden=32 (default) | 5,467 | 127× | 100% |
| decode `rnn` hidden=16 | 1,995 | 46× | 100% |
| decode `rnn` hidden=8 | 835 | 19× | 90% |
| decode `rnn` hidden=4 | 399 | 9× | 30% |
| encode MLP | 8,146 | 190× | 100% |
| parity MLP | 182 | 1.4× (vs 128 input patterns) | 100% |

Takeaways:

- **Only `parity` is near-minimal** (182 params vs 128 patterns). Its parity
  function has real XOR-like structure, so capacity ≈ problem size — also why
  architecture and seeds historically mattered for it.
- **The morse toys carry 20–190× the capacity the 43-row table needs.** The
  memorization knee is at hidden=16 (~46×); below it the net starts failing
  to memorize. Everything above is dead weight for a finite table.
- **RNN params do not scale with sequence length** — weight sharing across
  timesteps keeps the ~5.5k count constant whether the code is `E` (1 symbol)
  or `-.--.-` (6 symbols). That constant is what the MLP's growing padded
  input does *not* have.
- **Capacity ≠ quality on finite tasks**: all these get 100% by memorizing.
  Overcapacity will only matter (positively) on generalization, i.e. noisy
  symbols, jittered timing, and continuous unsplit streams — none of which
  these toys test yet.

## Support files

- `cli.py` — single CLI entry point (subcommands, one torch init per run).
- `common.py` — shared torch init: warnings,
  `--acceleration/--seed/--threads`, device banner, `model_summary`.
- `utils.py` — `NNet` (generic MLP with train loop, early stop) and
  `DatasetFeeder`.
- `morse_data.py` — morse table (43 entries) + pure text↔code helpers, used by
  the experiments and the `morse-translate` subcommand (no torch,
  import-safe).