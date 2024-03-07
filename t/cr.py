# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 14:00:58 2024

@author: kgavahi
"""

from Historic_Crypto import HistoricalData
import yfinance as yf


start_date = '1994-01-01-00-00'
end_date = '2024-01-18-00-00'
df_stk = yf.download('VOO', start_date[:10], end_date[:10])
df_stk.columns = [x.lower() for x in df_stk.columns]

# df_btc = HistoricalData('BTC-USD', 86400,
#                         start_date, 
#                         end_date).retrieve_data()
# df_eth = HistoricalData('ETH-USD', 86400,
#                         start_date, 
#                         end_date).retrieve_data()

capital_stk = 475.33*12*30
capital_btc = 300*12*5
capital_eth = 300*12*5




def alwaysBuy(df, capital):
    
    curretn_price = df['close'][-1]
    buy_price = capital/len(df)
    print('buy price daily', f'${buy_price:,.2f}')
    print('capital', f'${buy_price*len(df):,.2f}')
    

    df['p95'] = curretn_price/((df['high']+df['low'])/2) * buy_price
    


    
    tot_return = df['p95'].sum()

    
    return tot_return

def alwaysBuyWeekly(df, capital):
    
    curretn_price = df['close'][-1]
    buy_price = capital/len(df) * 7
    print('buy price weekly', f'${buy_price:,.2f}')
    
    df['p95'] = curretn_price/((df['high']+df['low'])/2) * buy_price
    
    tot_return = df[::7]['p95'].sum()
    
    return f'${tot_return:,.2f}'

print('\n')
print('VTI')
r = alwaysBuy(df_stk, capital_stk)
print('Return', f'${r:,.2f}')
print(f'{r/capital_stk:.2f}')

# print('\n')
# print('BTC')
# r = alwaysBuy(df_btc, capital_btc)
# print('Return', f'${r:,.2f}')
# print(f'{r/capital_btc:.2f}')

# print('\n')
# print('ETH')
# r = alwaysBuy(df_eth, capital_eth)
# print('Return', f'${r:,.2f}')
# print(f'{r/capital_eth:.2f}')



