import json
import sys
import os

def convert(py_file, ipynb_file):
    with open(py_file, 'r') as f:
        code = f.read()

    notebook = {
     "cells": [
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": code.splitlines(keepends=True)
      }
     ],
     "metadata": {
      "kernelspec": {
       "display_name": "Python 3",
       "language": "python",
       "name": "python3"
      },
      "language_info": {
       "codemirror_mode": {
        "name": "ipython",
        "version": 3
       },
       "file_extension": ".py",
       "mimetype": "text/x-python",
       "name": "python",
       "nbconvert_exporter": "python",
       "pygments_lexer": "ipython3",
       "version": "3.8"
      }
     },
     "nbformat": 4,
     "nbformat_minor": 4
    }

    with open(ipynb_file, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"Converted {py_file} to {ipynb_file}")

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
