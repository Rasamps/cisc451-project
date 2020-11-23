import os
import pandas as pd
import numpy as np


def main():
    df = pd.read_csv('data/master.csv', header = 0, index_col = None)
