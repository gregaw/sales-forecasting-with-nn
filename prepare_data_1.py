import pandas as pd
import numpy as np

input_file = "data/train.csv"
df = pd.read_csv(input_file, parse_dates=['Date'], dtype={'StateHoliday': np.str})

# TODO just one store for now
df = df[df.Store == 1]

# we want to start with earlier first
df.sort_values(by=['Date'], ascending=True, inplace=True)

train_size = int(0.7 * df.shape[0])
df[:train_size].to_csv('data/train1.csv', index=False)
df[train_size:].to_csv('data/eval1.csv', index=False)
