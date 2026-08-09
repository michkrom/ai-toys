#!/usr/bin/python3
# convert text to morse code
#
import os
import sys
import argparse

morseCode = [
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
    #(' ', '/'),
    #'\n', '//'
]

mapC2M = {key: value for key, value in morseCode}
mapM2C = {key: value for value, key in morseCode}


def toMorseCode(c):
    """Return the morse code for a char, or '' for unknown chars."""
    return mapC2M.get(c.upper(), '')


def filter(data, verbose):
    for line in data:
        for c in line:
            if verbose:
                yield c
                yield ' '
            yield toMorseCode(c)
            yield '\n' if verbose else ' '

def main():
    parser = argparse.ArgumentParser(description='Morse code translator')
    parser.add_argument(
        'filename',
        nargs='?',
        type=argparse.FileType('r'),
        help='Path to the input file (default: pipe text via stdin)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Print details about character processing (optional)',
    )
    args = parser.parse_args()

    # Open file object if provided, otherwise use stdin
    if args.filename:
        data = args.filename
    elif sys.stdin.isatty():
        # Interactive terminal with nothing piped in: there is no input to
        # translate, so error out with the usage instead of waiting forever.
        parser.error(
            'no input: pass a file argument or pipe text on stdin, '
            'e.g.  echo SOS | morseCode.py'
        )
    else:
        data = sys.stdin

    # Process one line at a time and flush after each, so piped/interactive
    # output appears immediately (non-verbose mode emits no newline).
    try:
        for line in data:
            for c in filter([line], args.verbose):
                sys.stdout.write(c)
            sys.stdout.flush()
    except BrokenPipeError:
        # e.g. `morseCode.py file | head`: exit quietly instead of a traceback
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == '__main__':
    main()
