#!/usr/bin/env python
# coding: utf-8

# # Current TD Portfolio

# HTML('''<script>
# code_show=true; 
# function code_toggle() {
#  if (code_show){
#  $('div.input').hide();
#  } else {
#  $('div.input').show();
#  }
#  code_show = !code_show
# } 
# $( document ).ready(code_toggle);
# </script>
# The raw code for this IPython notebook is by default hidden for easier reading.
# To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

# ## Fund Summary

# In[84]:


import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\private_data'
from IPython.display import Markdown as md
from IPython.display import display, HTML
import xlrd
from yahoo_fin.stock_info import *
from yahoo_fin import stock_info as si

import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Pastel1_7
import altair as alt


# In[3]:


# path = os.getcwd()
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
lstyear = str((datetime.strptime(td, '%m/%d/%Y') + timedelta(days = -365)).strftime('%m/%d/%Y'))
lstmon = str((datetime.strptime(td, '%m/%d/%Y') + timedelta(days = -90)).strftime('%m/%d/%Y'))
tmr = str((datetime.strptime(td, '%m/%d/%Y') + timedelta(days = 1)).strftime('%m/%d/%Y'))



#stock_tickers = stock_smr[(stock_smr.products == 'STOCK') | (stock_smr.products == 'ETF')]["ticker"].unique()
stock_tickers = stock_smr[stock_smr.products != 'CASH']["ticker"].unique()

df=pd.DataFrame()
for sticker in stock_tickers:
    data = si.get_data(sticker, start_date = lstmon, end_date = tmr)
    df = df.append(data)
df['date'] = df.index
curr_cadusd = pd.to_numeric(si.get_data('CADUSD=X', start_date = td, end_date = tmr).sort_index(ascending=False).head(1).adjclose)[0]

from pandasql import sqldf, load_meat, load_births
pysqldf = lambda q: sqldf(q, globals())

stock_curr_price = df[df.adjclose.notnull()].sort_values(by = ['date'], ascending=False).groupby('ticker').head(3)
stock_curr_price['rank'] = stock_curr_price.groupby('ticker')['date'].rank(method = 'first', ascending = False)

query = """
    SELECT 
        A.*, B.adjclose as today_price, C.adjclose as lastday_price, 
        B.adjclose / A.cost - 1 as today_roe, 
        B.adjclose / C.adjclose - 1 as today_ir,
        round((B.adjclose - A.cost) * A.share) as profit,
        round(case when A.ticker = 'CASH' then A.cost else B.adjclose * A.share end) as curr_value
    FROM 
        stock_smr A
    LEFT JOIN 
        stock_curr_price B on A.ticker = B.ticker and B.rank = 1
    LEFT JOIN 
        stock_curr_price C on A.ticker = C.ticker and C.rank = 2;
    """

stock_curr_smr = pysqldf(query)
display(stock_curr_smr)
stock_curr_smr['currency_index'] = [1 if x == 'CAD' else curr_cadusd for x in stock_curr_smr['currency']]
stock_curr_smr['curr_value_CAD'] = stock_curr_smr.curr_value / stock_curr_smr.currency_index

profit_CAD = round(stock_curr_smr[stock_curr_smr.category != 'CASH'][stock_curr_smr.currency == 'CAD']['profit'].sum())
profit_USD = round(stock_curr_smr[stock_curr_smr.category != 'CASH'][stock_curr_smr.currency == 'USD']['profit'].sum())
curr_fund_CAD = round(profit_CAD + initial_fund_CAD)
curr_fund_USD = round(profit_USD + initial_fund_USD)
total_curr_fund_CAD = curr_fund_CAD + round(curr_fund_USD / curr_cadusd)
total_ini_fund_CAD = round(initial_fund_CAD + initial_fund_USD / curr_cadusd)
cash_CAD = round(stock_curr_smr[stock_curr_smr.category == 'CASH'][stock_curr_smr.currency == 'CAD']['cost'].sum())
cash_USD = round(stock_curr_smr[stock_curr_smr.category == 'CASH'][stock_curr_smr.currency == 'USD']['cost'].sum())
investment_CAD = curr_fund_CAD - cash_CAD
investment_USD = curr_fund_USD - cash_USD
ivt_CAD_pct = (investment_CAD / curr_fund_CAD)
ivt_USD_pct = (investment_USD / curr_fund_USD)
ivt_pct = ((investment_CAD + investment_USD / curr_cadusd) / total_curr_fund_CAD)
inv_roc_CAD = profit_CAD / investment_CAD
inv_roc_USD = profit_USD / investment_USD
inv_roc_total = (profit_CAD + profit_USD / curr_cadusd) / (investment_CAD + investment_USD / curr_cadusd)


# In[28]:


print("Initial CAD fund: ", initial_fund_CAD, ". Initial USD fund: ", initial_fund_USD, 
      "\nCurrent CAD fund: ", curr_fund_CAD, ". Current USD fund:", curr_fund_USD,
      "\nTotal profit in CAD: ", profit_CAD, ". Total profit in USD: ", profit_USD,
      "\nROC in CAD: ", "{:.1%}".format(inv_roc_CAD), ". ROC in USD: ", "{:.1%}".format(inv_roc_USD), ". ROC overall: ", "{:.1%}".format(inv_roc_total),
      "\nCurrent cash in CAD: ", cash_CAD, ". Current cash in USD: ", cash_USD,
      "\nCurrent investment in CAD: ", investment_CAD, ".Current investment in USD: ", investment_USD,
     "\nInitial fund in CAD: ", total_ini_fund_CAD, ". Current fund in CAD: ", total_curr_fund_CAD,
     "\nInvestment % in CAD: ", "{:.1%}".format(ivt_CAD_pct), ". Investment % in USD: ", "{:.1%}".format(ivt_USD_pct), ". Investment % total: ", "{:.1%}".format(ivt_pct))


# ## Fund Category Distribution

# In[29]:


import matplotlib.pyplot as plt

category_data_smr = pd.DataFrame(stock_curr_smr.groupby('category')['curr_value_CAD'].agg('sum').round(0))
display(HTML(category_data_smr.to_html()))

category_data_smr['category'] = category_data_smr.index

my_circle=plt.Circle( (0,0), 0.5, color='white')
# plot link: https://python-graph-gallery.com/161-custom-matplotlib-donut-plot/
from palettable.colorbrewer.qualitative import Pastel1_7
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))
plt.pie(category_data_smr.curr_value_CAD, 
        labels=category_data_smr.category, 
        colors=Pastel1_7.hex_colors, 
        autopct='%1.1f%%',
        wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# ## Fund Products Distribution

# In[30]:


product_data_smr = pd.DataFrame(stock_curr_smr.groupby('products')['curr_value_CAD'].agg('sum'))
display(HTML(product_data_smr.to_html()))
product_data_smr['products'] = product_data_smr.index

my_circle=plt.Circle( (0,0), 0.5, color='white')
# plot link: https://python-graph-gallery.com/161-custom-matplotlib-donut-plot/
from palettable.colorbrewer.qualitative import Pastel1_7
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))
plt.pie(product_data_smr.curr_value_CAD, 
        labels=product_data_smr.products, 
        colors=Pastel1_7.hex_colors, 
        autopct='%1.1f%%',
        wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# ## Current Holding Market Watch

# In[14]:


# generate plots: volume analysis
import altair as alt
alt.Chart(df).mark_line().encode(x='date',y='volume', column = 'ticker')
alt.Chart(df).mark_line().encode(x='date',y='volume', color = 'ticker:N')


# In[16]:


import matplotlib.pyplot as plt

df.groupby('ticker').plot(x = 'date', y = 'volume')

fig, ax = plt.subplots(figsize=(15,7))
df.groupby('ticker')['volume'].plot(legend=True)


# ## Watch List

# In[82]:


watchlist = pd.read_excel(os.path.join(path,file), sheet_name = 'watchlist')
watchlist_ticker = watchlist["ticker"].unique()
watchlist_df=pd.DataFrame()
for sticker in watchlist_ticker:
    watchlist_df = watchlist_df.append(si.get_data(sticker, start_date = lstyear, end_date = tmr))
watchlist_df['date'] = watchlist_df.index
watchlist_df.head(3)


# In[85]:


alt.Chart(watchlist_df).mark_line().encode(x='date',y='volume', color = 'ticker:N')


# In[97]:


import yahooquery
from yahooquery import Ticker

def get_etf_holding(ticker):
    t = Ticker(ticker)
    print('===============================',ticker,'=================================')
    print()
    print('-------------------------- Key Factor ----------------------------')
    print(pd.DataFrame(t.key_stats[ticker], index=[0]).T.set_axis(['Value'], axis=1, inplace=False))
    print('-------------------------- Fund Sector ----------------------------')
    fund_sector = t.fund_sector_weightings
    if len(fund_sector) >1:
        fund_sector.insert(0,'Fund Sector',fund_sector.index)
        fund_sector = fund_sector.reset_index(drop=True)
        fund_sector.columns = ['Fund Sector','Percentage']
        fund_sector = fund_sector.sort_values(by='Percentage', ascending=False)
    print(fund_sector)
    print('-------------------------- Holdings ----------------------------')
    if type(t.fund_holding_info[ticker]) != str:
        print(pd.DataFrame(t.fund_holding_info[ticker]['holdings']))
    print('-------------------------- Returns ----------------------------')
    if type(t.fund_performance[ticker]) != str:
        print(pd.DataFrame(t.fund_performance[ticker]['performanceOverview'], index=[0]).T.set_axis(['Value'], axis=1, inplace=False))
        print()
        print(pd.DataFrame(t.fund_performance[ticker]['trailingReturns'], index=[0]).T.set_axis(['Value'], axis=1, inplace=False))
        print()
        print(pd.DataFrame(t.fund_performance[ticker]['annualTotalReturns']['returns']))
    print('-------------------------- Risk Factors ----------------------------')
    if type(t.fund_performance[ticker]) != str:
        print(pd.DataFrame(t.fund_performance[ticker]['riskOverviewStatistics']['riskStatistics']))
    print('=========================================================================')
 
for i in watchlist_ticker:
    get_etf_holding(i)


# In[98]:


get_etf_holding('0P00016N6T.TO')

