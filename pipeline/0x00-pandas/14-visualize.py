#!/usr/bin/env python3
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file


def open_d(array):
    return array[0]


def close_d(array):
    return array[-1]


df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.loc[df['Timestamp'] >= 1483228800]
df = df.drop(columns=['Weighted_Price'])
df = df.rename(columns={'Timestamp': 'Date'})
df['Date'] = pd.to_datetime(arg=df['Date'], unit='s')
# df['year_month'] = df['Date'].dt.to_period('M')
df.set_index('Date', inplace=True)
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


df_d = pd.DataFrame()
df_d['Open'] = df.Open.resample('D').apply(open_d)
df_d['High'] = df.High.resample('D').max()
df_d['Low'] = df.Low.resample('D').min()
df_d['Close'] = df.Close.resample('D').apply(close_d)
df_d['Volume_(BTC)'] = df['Volume_(BTC)'].resample('D').sum()
df_d['Volume_(Currency)'] = df['Volume_(Currency)'].resample('D').sum()


df_d.plot()
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 14000000))

plt.show()
