# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:58:06 2020

@author: luoyan011
"""
import os
import pandas as pd
os.chdir('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\StockDynamicReport')

import xlrd
path = os.getcwd()
file = 'stock_smr.xlsx'
sheet = 'holding'
xlsf = pd.ExcelFile(os.path.join(path, file))
stock_smr = pd.read_excel(os.path.join(path,file), sheet_name = sheet)
initial_fund_CAD = 35360.52
initial_fund_USD = 17540.96
current_cash_CAD = stock_smr[(stock_smr.products == 'CASH') & (stock_smr.currency == 'CAD')]['cost']
current_cash_USD = stock_smr[(stock_smr.products == 'CASH') & (stock_smr.currency == 'USD')]['cost']


from datetime import datetime, timedelta
td = str((datetime.today() - timedelta(days = max(1, (datetime.today().weekday() + 6) % 7 - 3))).strftime('%m/%d/%Y'))
lstmon = str((datetime.strptime(td, '%m/%d/%Y') + timedelta(days = -90)).strftime('%m/%d/%Y'))
tmr = str((datetime.strptime(td, '%m/%d/%Y') + timedelta(days = 1)).strftime('%m/%d/%Y'))


from yahoo_fin.stock_info import *
from yahoo_fin import stock_info as si

#stock_tickers = stock_smr[(stock_smr.products == 'STOCK') | (stock_smr.products == 'ETF')]["ticker"].unique()
stock_tickers = stock_smr[stock_smr.products != 'CASH']["ticker"].unique()

df=pd.DataFrame()
for sticker in stock_tickers:
    data = si.get_data(sticker, start_date = lstmon, end_date = tmr)
    df = df.append(data)
df['date'] = df.index

curr_cadusd = pd.to_numeric(si.get_data('CADUSD=X', start_date = td, end_date = tmr).sort_index(ascending=False).head(1).adjclose)[0]
#%% Plot

#https://stackoverflow.com/questions/33150510/how-to-create-groupby-subplots-in-pandas/33152131
import altair as alt


alt.Chart(df).mark_line().encode(x='date',y='volume', column = 'ticker')
alt.Chart(df).mark_line().encode(x='date',y='volume', color = 'ticker:N')


import matplotlib.pyplot as plt

df.groupby('ticker').plot(x = 'date', y = 'volume')

fig, ax = plt.subplots(figsize=(15,7))
df.groupby('ticker')['volume'].plot(legend=True)

#%% plortfolio calculation
from pandasql import sqldf, load_meat, load_births
pysqldf = lambda q: sqldf(q, globals())

stock_curr_price = df[df.adjclose.notnull()].sort_values(by = ['date'], ascending=False).groupby('ticker').head(3)
stock_curr_price['rank'] = stock_curr_price.groupby('ticker')['date'].rank(method = 'first', ascending = False)

query = """
    SELECT 
        A.*, B.adjclose as today_price, C.adjclose as lastday_price, 
        B.adjclose / A.cost - 1 as roe, B.adjclose / C.adjclose - 1 as today_ir,
        (B.adjclose - A.cost) * A.share as profit
    FROM 
        stock_smr A
    LEFT JOIN 
        stock_curr_price B on A.ticker = B.ticker and B.rank = 1
    LEFT JOIN 
        stock_curr_price C on A.ticker = C.ticker and C.rank = 2;
    """

stock_curr_smr = pysqldf(query)

curr_fund_CAD = stock_curr_smr[stock_curr_smr.category != 'CASH'][stock_curr_smr.currency == 'CAD']['profit'].sum() + initial_fund_CAD 
curr_fund_USD = stock_curr_smr[stock_curr_smr.category != 'CASH'][stock_curr_smr.currency == 'USD']['profit'].sum() + initial_fund_USD 
