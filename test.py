import pandas as pd

data = pd.read_csv("log.csv").iloc[1:]

data = data.set_index(['time']) 

data.index = pd.to_datetime(data.index) 

groups = data.resample('5s')
    
split_dfs = [group for _, group in groups]

for i in split_dfs:
    print(i)

