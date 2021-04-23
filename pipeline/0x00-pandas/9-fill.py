#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.drop(columns='Weighted_Price', inplace=True)
df.loc[:, 'Close'].fillna(method='ffill', inplace=True)
df.loc[:, 'Open'].fillna(value=df.loc[:, 'Close'].shift(periods=1,
                                                        axis='index'),
                         inplace=True)
df.loc[:, 'High'].fillna(value=df.loc[:, 'Close'].shift(periods=1,
                                                        axis='index'),
                         inplace=True)
df.loc[:, 'Low'].fillna(value=df.loc[:, 'Close'].shift(periods=1,
                                                       axis='index'),
                        inplace=True)
df.loc[:, 'Volume_(BTC)'].fillna(value=0, inplace=True)
df.loc[:, 'Volume_(Currency)'].fillna(value=0, inplace=True)

print(df.head())
print(df.tail())
