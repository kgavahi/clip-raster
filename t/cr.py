# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 14:00:58 2024

@author: kgavahi
"""

from Historic_Crypto import HistoricalData
import pandas as pd
df = HistoricalData('BTC-USD',86400,'2020-01-01-00-00').retrieve_data()

capital = 100/7 * len(df)
curretn_price = df['close'][-1]



def alwaysBuy(df, capital, curretn_price):
    

    buy_price = 10390/len(df)
    print('buy price alwaysBuy', buy_price)
    print('capital', buy_price*len(df))
    

    df['p95'] = curretn_price/((df['high']+df['low'])/2) * buy_price

    
    tot_return = df['p95'].sum()
    
    return tot_return

def alwaysBuyWeekly(df, capital, curretn_price):
    

    buy_price = 100
    print('buy price alwaysBuy', buy_price)
    
    #df['p95'] = curretn_price/(df['open']) * buy_price
    df['p95'] = curretn_price/((df['high']+df['low'])/2) * buy_price
    #df['p95'] = curretn_price/(df['high']) * buy_price
    
    tot_return = df[::7]['p95'].sum()
    
    return tot_return

print('\n')
#print('capital', capital)
print('alwaysBuy:',alwaysBuy(df, capital, curretn_price))
#print('alwaysBuyWeekly:',alwaysBuyWeekly(df, capital, curretn_price))


