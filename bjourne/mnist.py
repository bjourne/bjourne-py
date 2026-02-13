# Copyright (C) 2026 Björn Lindqvist <bjourne@gmail.com>
#
# Dirt simple module for downloading and loading MNIST.
from gzip import open as gzip_open
from pathlib import Path
from urllib.request import urlretrieve
from struct import unpack
import numpy as np

URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

FILENAMES = [
    "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
]

def download(url, path):
    if not path.is_file():
        urlretrieve(url, path)

def load_images(path):
    with gzip_open(path, "rb") as f:
        magic, n, r, c = unpack(">IIII", f.read(16))
        assert magic == 2051
        buf = f.read()
        return np.frombuffer(buf, np.uint8).reshape(n, r, c)

def load_labels(path):
    with gzip_open(path, "rb") as f:
        magic, n = unpack(">II", f.read(8))
        assert magic == 2049
        buf = f.read()
        return np.frombuffer(buf, np.uint8)

def load(path):
    path = Path(path)
    path.mkdir(parents = True, exist_ok = True)
    for fname in FILENAMES:
        download(URL + fname, path / fname)
    tr_x = load_images(path / FILENAMES[0])
    tr_y = load_labels(path / FILENAMES[1])
    te_x = load_images(path / FILENAMES[2])
    te_y = load_labels(path / FILENAMES[3])
    return (tr_x, tr_y), (te_x, te_y)

if __name__ == "__main__":
    from shutil import get_terminal_size
    np.set_printoptions(
        precision=2,
        linewidth = get_terminal_size(fallback=(120,20)).columns
    )
    load("data")
