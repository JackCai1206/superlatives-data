import pandas as pd
import numpy as np
from collections import defaultdict
import math 

data = ['question' 'name']
df = pd.read_csv(r'gise_pt2.csv')

# print(df.iloc[:,1])
for i in range(2, len(df.columns)):
    col = df.iloc[:,i]
    total = 0
    votes = defaultdict(int)
    for name in col:
        if i == 3:
            print(type(name))
        if(isinstance(name,float)):
            continue
        votes[name] += 1
        total += 1
    if i == 3:
        print(votes)
# for i in df.columns:

