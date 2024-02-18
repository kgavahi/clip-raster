# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 15:21:51 2024

@author: kgavahi
"""

import yfinance as yf
import pandas as pd

date = '2020-01-01'
end_date = '2024-02-17'
# Get the data for the stock AAPL
df = yf.download('VTI',date, end_date)

from datetime import datetime
date_format = "%Y-%m-%d"

a = datetime.strptime(date, date_format)
b = datetime.strptime(end_date, date_format)

delta = b - a

print(delta.days) 


capital = 100/7 * delta.days
curretn_price = df['Close'][-1]




def alwaysBuy(df, capital, curretn_price):
    

    buy_price = 10
    print('capital', len(df)*10)
    
    df['p95'] = curretn_price/((df['High']+df['Low'])/2) * buy_price
    
    tot_return = df['p95'].sum()
    
    return tot_return


print('alwaysBuy', alwaysBuy(df, capital, curretn_price))

