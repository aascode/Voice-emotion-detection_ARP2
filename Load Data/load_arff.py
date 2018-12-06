import pandas as pd
from arff2pandas import a2p

with open('output.arff') as f:
    df = a2p.load(f)
    print(df)