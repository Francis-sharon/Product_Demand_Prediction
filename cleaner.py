import pandas as pd

df = pd.read_csv("raw.csv")

df['diff'] = df['base'] - df['total']
x = 1

df['demand'] = df['diff'].apply(lambda x: 1 if x > 15 else 0)

df.to_csv('source.csv')

