#!/usr/bin/python
# convert text to morse code
#
import sys
import argparse
from utils import DatasetFeeder, NNet

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

parser = argparse.ArgumentParser(description='Morse code translator')
parser.add_argument(
    'filename',
    nargs='?',
    type=argparse.FileType('r'),
    help='Path to the input file (default, stdin)'
)
parser.add_argument(
    '--verbose',
    action='store_true',
    default=False,
    help='Print details about character processing (optional)',
)
args = parser.parse_args()


def toMorseCode(c):
    c = c.upper()
    return mapC2M[c if c in morseCode else ' ']


def filter(data, verbose):
    for line in data:
        for c in line:
            if verbose:
                yield c
                yield ' '
            yield toMorseCode(c)
            yield '\n' if verbose else ' '

def main():
    # Open file object if provided, otherwise use stdin
    if args.filename:
        data = args.filename
    else:
        data = sys.stdin

    # Write filtered data to stdout (one character at a time)
    for c in filter(data, args.verbose):
        sys.stdout.write(c)


if __name__ == '__main__':
    main()
