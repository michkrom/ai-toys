#!/usr/bin/env python3
"""
cli.py — single entry point for all torch toys (one torch init per run).

Subcommands are named <domain>-<action> so the set can grow:

    ./cli.py morse-encode    [--epochs N] [--samples N] [--acceleration auto|cpu|gpu|gpu:N] [--seed N] [--threads N]
    ./cli.py morse-decode    [same flags] [--model mlp|gru]
    ./cli.py parity-learn    [same flags]
    ./cli.py morse-translate [--verbose] [FILE]   # non-ML reference translator

Any ML experiment can also still be run directly, e.g. ./morseDecode.py,
but the CLI routes everything through one shared torch init (common.py).
"""
import argparse
import sys

import common
import morse_data
import morse
import parity
import morseDecode


def run_translate(args):
    if args.file:
        data = args.file
    elif sys.stdin.isatty():
        sys.stderr.write(
            "cli.py morse-translate: no input: pass a file argument or pipe "
            "text on stdin, e.g.  echo SOS | ./cli.py morse-translate\n"
        )
        sys.exit(2)
    else:
        data = sys.stdin
    morse_data.handle_pipe_errors(lambda: morse_data.translate_stream(data, args.verbose))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cli.py", description="run the torch ML toys / tools"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, module, help_text in (
        ("morse-encode", morse, "ML: learn char -> morse code"),
        ("morse-decode", morseDecode, "ML: learn morse code -> char (--model mlp|gru)"),
        ("parity-learn", parity, "ML: learn the parity bit of a 7-bit byte"),
    ):
        sp = sub.add_parser(name, help=help_text)
        common.add_torch_args(sp)      # one shared torch init for all ML runs
        module.add_args(sp)
        sp.set_defaults(fn=module.run)

    sp = sub.add_parser("morse-translate",
                        help="plain text<->morse reference translator (no ML)")
    sp.add_argument("file", nargs="?", type=argparse.FileType("r"),
                    help="input file (default: pipe text via stdin)")
    sp.add_argument("--verbose", action="store_true",
                    help="print each input char before its code")
    sp.set_defaults(fn=run_translate)

    args = parser.parse_args(argv)
    if hasattr(args, "acceleration"):   # ML subcommands only
        common.finish_args(args)
        print(f"[cli] run={args.cmd} device={common.describe_device(args.device)}"
              + (f" seed={args.seed}" if args.seed is not None else " seed=<default>")
              + (f" threads={args.threads}" if args.threads is not None else ""))
    else:
        print(f"[cli] run={args.cmd} (no torch needed)")
    args.fn(args)


if __name__ == "__main__":
    main()