# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


def retrieve_hp(ticker):
    
    base_url = f"https://finance.yahoo.com/quote/{ticker}/history?period1=946684800&period2=1703980800"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}
    response = requests.get(base_url,headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'table svelte-ewueuo'})

    rows = table.find_all('tr')

    column_names = [th.text.strip() for th in rows[0].find_all('th')]

    data = []
    for i in range(1,len(rows)):
        row = rows[i]
        cols = row.find_all('td')
        cols_text = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols_text if ele])

    df = pd.DataFrame(data, columns=column_names)
    
    df.rename(columns={'Adj Close       Adjusted close price adjusted for splits and dividend and/or capital gain distributions.': 'Price'}, inplace=True)
    df = df[['Date', 'Price']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by=['Date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    return df

retrieve_hp('AAPL')
ticker_list = ['AMZN', 'AAPL', 'GOOGL', 'TSLA', 'MSFT', 'NVDA']
df_ff = pd.read_csv('Factors.csv')
df_ff['Date'] = pd.to_datetime(df_ff['Date'], format='%Y%m%d')
df_ff

params_Mkt = {}
params_SMB = {}
params_HML = {}

for ticker in ticker_list:
    
    # Get stock price data from Yahoo Finance
    df = retrieve_hp(ticker)

    # Calculate return
    # shift(1) shifts the series by one row
    df['Return'] = (df['Price'] - df['Price'].shift(1)) / df['Price'].shift(1) * 100

    # merge the stock price data with the factor data
    df = pd.merge(df, df_ff, on='Date')
    # calculate excess return of the stock
    df['Excess_Return'] = df['Return'] - df['RF']

    # run regression
    df = df[['Excess_Return', 'Mkt-RF', 'SMB', 'HML']].dropna()
    y = df['Excess_Return']
    X = df[['Mkt-RF', 'SMB', 'HML']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # store the parameters
    params_Mkt[ticker] = model.params['Mkt-RF']
    params_SMB[ticker] = model.params['SMB']
    params_HML[ticker] = model.params['HML']
    
    
plt.figure(figsize=(9,3))

plt.subplot(1,3,1)
plt.title('Market Premium')
plt.bar(params_Mkt.keys(),  params_Mkt.values())
plt.xticks(rotation = 35)

plt.subplot(1,3,2)
plt.title('SMB')
plt.bar(params_SMB.keys(),  params_SMB.values())
plt.axhline(0, color='gray', linewidth=0.8)
plt.xticks(rotation = 35)

plt.subplot(1,3,3)
plt.title('HML')
plt.bar(params_HML.keys(),  params_HML.values())
plt.xticks(rotation = 35)

plt.show()    
