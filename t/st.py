# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 15:21:51 2024

@author: kgavahi
"""

import yfinance as yf
import pandas as pd

date = '2017-12-18'
end_date = '2024-02-18'
# Get the data for the stock AAPL
df = yf.download('VTI',date, end_date)
print('\n')


capital = 2000
curretn_price = df['Close'][-1]




def alwaysBuy(df, capital, curretn_price):
    

    buy_price = capital/len(df)
    print('buy_price', f'${buy_price:,.2f}')
    print('capital', f'${capital:,.2f}')
    
    df['p95'] = curretn_price/((df['High']+df['Low'])/2) * buy_price
    
    tot_return = df['p95'].sum()
    
    return tot_return

r = alwaysBuy(df, capital, curretn_price)
print('Return', f'${r:,.2f}')
print(f'{r/capital:.2f}')

