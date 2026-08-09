#!/usr/bin/env python3
"""
morse_data.py — morse code table and text<->code helpers.

Shared data + pure functions used by the ML experiments (encode/decode) and
by the non-ML translate command in cli.py. No CLI, no torch, no side effects
on import.
"""
import os
import sys

# (character, morse code) for A-Z, 0-9 and punctuation; 43 entries
MORSE_CODES = [
    ('A', '.-'),
    ('B', '-...'),
    ('C', '-.-.'),
    ('D', '-..'),
    ('E', '.'),
    ('F', '..-.'),
    ('G', '--.'),
    ('H', '....'),
    ('I', '..'),
    ('J', '.---'),
    ('K', '-.-'),
    ('L', '.-..'),
    ('M', '--'),
    ('N', '-.'),
    ('O', '---'),
    ('P', '.--.'),
    ('Q', '--.-'),
    ('R', '.-.'),
    ('S', '...'),
    ('T', '-'),
    ('U', '..-'),
    ('V', '...-'),
    ('W', '.--'),
    ('X', '-..-'),
    ('Y', '-.--'),
    ('Z', '--..'),
    ('1', '.----'),
    ('2', '..---'),
    ('3', '...--'),
    ('4', '....-'),
    ('5', '.....'),
    ('6', '-....'),
    ('7', '--...'),
    ('8', '---..'),
    ('9', '----.'),
    ('0', '-----'),
    (',', '--..--'),
    ('.', '.-.-.-'),
    ('?', '..--..'),
    ('/', '-..-.'),
    ('-', '-....-'),
    ('(', '-.--.'),
    (')', '-.--.-'),
]

char_to_code = {ch: code for ch, code in MORSE_CODES}
code_to_char = {code: ch for ch, code in MORSE_CODES}


def to_morse_code(ch):
    """morse for a char, or '' for unknown chars."""
    return char_to_code.get(ch.upper(), '')


def stream_line(line, verbose=False):
    """yield the morse rendering of one input line, char by char."""
    for c in line:
        if verbose:
            yield c
            yield ' '
        yield to_morse_code(c)
        yield '\n' if verbose else ' '


def translate_stream(lines, verbose=False, out=None):
    """translate an iterable of lines to stdout (or the given stream)."""
    out = out or sys.stdout
    for line in lines:
        for c in stream_line(line, verbose):
            out.write(c)
        out.flush()


def handle_pipe_errors(fn, out=None):
    """run fn() with quiet exits on Ctrl-C / broken pipes."""
    try:
        fn()
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, (out or sys.stdout).fileno())
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)