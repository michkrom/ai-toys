#!/usr/bin/env python3
"""
common.py — shared torch init for all experiments in this playground.

One place for the boring glue every run needs:
  * suppress noisy warnings (incl. torch CUDA deprecation noise)
  * pick the compute acceleration: auto | cpu | gpu | gpu:N (via --acceleration)
  * optional single seed for python+torch (reproducibility)
  * optional thread clamp for CPU runs (useful on shared/loaded machines)
"""
import argparse
import random
import re
import warnings

# Silence (incl. torch's CUDA/pynvml deprecation warning) before torch import
warnings.filterwarnings("ignore")

import torch

from torch import nn


def model_summary(model):
    """(topology, param_count) for an nn.Module -- printed when running --model."""
    parts = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            parts.append(f"Linear {m.in_features}->{m.out_features}")
        elif isinstance(m, nn.GRU):
            parts.append(f"GRU {m.input_size}->{m.hidden_size} x{m.num_layers}")
        elif isinstance(m, nn.Embedding):
            parts.append(f"Embedding {m.num_embeddings}->{m.embedding_dim}")
    return ", ".join(parts), sum(p.numel() for p in model.parameters())


def acceleration_type(value):
    """Validate --acceleration values: auto | cpu | gpu | gpu:N | cuda:N."""
    if value in ("auto", "cpu", "gpu"):
        return value
    match = re.fullmatch(r"(?:gpu|cuda):?(\d+)", value)
    if match:
        return f"gpu:{match.group(1)}"
    raise argparse.ArgumentTypeError(
        f"'{value}' is not valid; use auto, cpu, gpu, or gpu:N"
    )


def resolve_acceleration(accel="auto"):
    """'auto' -> cuda:0 if available else cpu; also cpu, gpu, gpu:N (or cuda:N)."""
    if accel == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if accel == "cpu":
        return torch.device("cpu")
    index = 0 if accel == "gpu" else int(accel.split(":")[1])
    if not torch.cuda.is_available():
        raise SystemExit(f"acceleration '{accel}' requested but torch has no CUDA")
    if index >= torch.cuda.device_count():
        raise SystemExit(
            f"GPU {index} requested but only {torch.cuda.device_count()} present "
            "(see nvidia-smi)"
        )
    return torch.device(f"cuda:{index}")


def init_torch(acceleration="auto", seed=None, threads=None):
    """Prepare torch for one experiment run; returns the resolved device."""
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    if threads is not None:
        torch.set_num_threads(threads)
    return resolve_acceleration(acceleration)


def describe_device(device):
    """Human-readable device string: 'cuda:0 (NVIDIA RTX A2000 12GB)' or 'cpu'."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device.index or 0)
        return f"{device} ({name})"
    return str(device)


def add_torch_args(parser):
    """Standard CLI flags for every ML experiment (acceleration / seed / threads)."""
    parser.add_argument(
        "--acceleration", "--accel", dest="acceleration",
        type=acceleration_type, default="auto",
        help="compute acceleration: auto (default: GPU when available, else CPU) | "
             "cpu | gpu | gpu:N (select GPU number N, see nvidia-smi)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="seed python+torch RNGs (default: per-experiment defaults)",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="limit torch CPU threads (e.g. 2 on a shared machine)",
    )


def finish_args(args):
    """Call after parse_args: resolve --acceleration into args.device."""
    args.device = init_torch(args.acceleration, args.seed, args.threads)
    return args