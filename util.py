"""
File: util.py
Author: Hongyu Li (hongyul)
"""

import numpy as np

# Read a file
# filename is the path of the file, string type
# returns the content as a string
def readFile(filename, mode = "rt"):
    # rt stands for "read text"
    fin = contents = None
    try:
        fin = open(filename, mode)
        contents = fin.read()
    finally:
        if (fin != None): fin.close()
    return contents


# Write 'contents' to the file
# 'filename' is the path of the file, string type
# 'contents' is of string type
# returns True if the content has been written successfully
def writeFile(filename, contents, mode = "wt"):
    # wt stands for "write text"
    fout = None
    try:
        fout = open(filename, mode)
        fout.write(contents)
    finally:
        if (fout != None): fout.close()
    return True

def load_symbols(path):
    symbols = readFile(path).strip().split("\n")
    word_idx = dict((c, i) for i, c in enumerate(symbols))
    return symbols, word_idx

def load_tags(path):
    tags = readFile(path).strip().split("\n")
    return tags

def load_emission(path):
    lines = readFile(path).strip().split("\n")
    emission = []
    for line in lines:
        tokens = line.strip().split()
        prob = [float(token.strip().split('%')[1]) for token in tokens[1:]]
        emission.append(prob)
    return np.array(emission)

def load_transition(path):
    lines = readFile(path).strip().split('\n')
    transition = []
    for line in lines:
        tokens = line.strip().split()
        prob = [float(token.strip().split(':')[1]) for token in tokens[1:]]
        transition.append(prob)
    return np.array(transition)

def load_priors(path):
    lines = readFile(path).strip().split('\n')
    return [float(line.strip().split(':')[1]) for line in lines]

def load_tweets(path):
    lines = readFile(path).strip().split('\n')
    return [line.strip().split() for line in lines]

def vectorize_sequence(seq, word_idx):
    return [word_idx.get(t, -1) for t in seq]