#!/usr/bin/env python
# coding: utf-8

# # Current TD Portfolio

# In[9]:


HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')


# ## Fund Summary

# In[33]:


import warnings
import numpy as np
warnings.filterwarnings('ignore')
import os
import pandas as pd
path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\private_data'
from IPython.display import Markdown as md
from IPython.display import display, HTML
import xlrd
from yahoo_fin.stock_info import *
from yahoo_fin import stock_info as si
from yahoo_fin import news
import timestring

import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Pastel1_7
import altair as alt
import plotly.express as px
import locale
locale.setlocale( locale.LC_ALL, '' ) #this used to format currency

import yahooquery
from yahooquery import Ticker
import yfinance as yf

# ===================================== Functions

# def get_stock_details(tickers, startdate, enddate):
#     output = pd.DataFrame()
#     for ticker in tickers:
#         tmp = si.get_data(ticker, start_date = startdate, end_date = enddate)
#         output = output.append(tmp)
#     output['date'] = output.index
#     return output
def get_stock_details(tickers, startdate, enddate):
    output = pd.DataFrame()
    for ticker in tickers:
        tmp = yf.Ticker(ticker).history(start = startdate, end = enddate)
        tmp['ticker'] = ticker
        output = output.append(tmp)
    output = output.reset_index()
    return output


def plot_pie_chart_dist(data, group_by_feature, dist_feature, optional_title = ''):
#     CSS = """
#     .output {
#         display: flex;
#         flex-direction: row;
#         flex-flow: wrap;
#     }
#     """

#     display(HTML('<style>{}</style>'.format(CSS)))
    # plot link: https://python-graph-gallery.com/161-custom-matplotlib-donut-plot/
    data_smr = pd.DataFrame(data.groupby(group_by_feature)[dist_feature].agg('sum').round(0))
#     display(HTML(data_smr.to_html()))
    data_smr[group_by_feature] = data_smr.index

    my_circle=plt.Circle( (0,0), 0.5, color='white')
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))

    total = sum(data_smr[dist_feature])
    def label_format(x):
        return('{:.1f}%\n({:.0f})'.format(x, total*x/100))
    plt.pie(data_smr[dist_feature], 
            labels=data_smr.index, 
            colors=Pastel1_7.hex_colors, 
            autopct=label_format,
            wedgeprops = { 'linewidth' : 15, 'edgecolor' : 'white' })
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    plt.title(optional_title + dist_feature + ' distribution by different ' + group_by_feature)
    plt.show()

    
def get_etf_holding(ticker, price_period = '2y'):

    t = Ticker(ticker)
    hist_price = t.history(period=price_period, interval='1d')
    hist_price = hist_price.reset_index()
    price = pd.melt(hist_price, id_vars=['date'], value_vars=['open', 'high', 'close', 'low', 'adjclose'])
    fig = px.line(price, x='date', y='value', color='variable',
                 title=ticker+' Historical Price')
    fig.show()
    fig = px.line(hist_price, x='date', y='volume', 
                 title=ticker+' Volume')
    fig.show()
    print('===============================',ticker,'=================================')
    print()
    print('-------------------------- Summary Details ----------------------------')
    smr_details = pd.DataFrame(t.summary_detail[ticker], index=[0]).T.set_axis(['Value'], axis=1, inplace=False)
    keep_cols = ['totalAssets', 'previousClose', 'open', 'fiftyDayAverage','fiftyDayAverage',
             'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'twoHundredDayAverage',
             'volume','averageVolume', 'averageVolume10days','averageDailyVolume10Day']
    smr_details = smr_details[smr_details.index.isin(keep_cols)]
    smr_details = smr_details.reindex(keep_cols)
    display(smr_details)
    print('-------------------------- Key Factor ----------------------------')
    key_factor = pd.DataFrame(t.key_stats[ticker], index=[0]).T.set_axis(['Value'], axis=1, inplace=False)
    key_factor = key_factor[key_factor.index.isin(['category','ytdReturn','fundFamily', 'fundInceptionDate', 'legalType'])]
    display(key_factor)
    print('-------------------------- Fund Sector ----------------------------')
    fund_sector = t.fund_sector_weightings
    if len(fund_sector) >1:
        fund_sector.insert(0,'Fund Sector',fund_sector.index)
        fund_sector = fund_sector.reset_index(drop=True)
        fund_sector.columns = ['Fund Sector','Percentage']
        fund_sector = fund_sector.sort_values(by='Percentage', ascending=False)
    display(fund_sector)
    print('-------------------------- Holdings ----------------------------')
    if type(t.fund_holding_info[ticker]) != str:
        display(pd.DataFrame(t.fund_holding_info[ticker]['holdings']))
    print('-------------------------- Returns ----------------------------')
    if type(t.fund_performance[ticker]) != str:
        display(pd.DataFrame(t.fund_performance[ticker]['performanceOverview'], index=[0]).T.set_axis(['Value'], axis=1, inplace=False))
        print()
        display(pd.DataFrame(t.fund_performance[ticker]['trailingReturns'], index=[0]).T.set_axis(['Value'], axis=1, inplace=False))
        print()
        display(pd.DataFrame(t.fund_performance[ticker]['annualTotalReturns']['returns']))
    print('-------------------------- Risk Factors ----------------------------')
    if type(t.fund_performance[ticker]) != str:
        display(pd.DataFrame(t.fund_performance[ticker]['riskOverviewStatistics']['riskStatistics']))
    print('=========================================================================')
                           


# In[2]:


#=========================== Load File
file = 'stock_smr.xlsx'
holding_tranc = 'holding_transaction'
cash_tranc = 'cash_transaction'

from datetime import datetime, timedelta
# td = str((datetime.today() - timedelta(days = max(1, (datetime.today().weekday() + 6) % 7 - 3))).strftime('%m/%d/%Y'))
# lstyear = str((datetime.strptime(td, '%m/%d/%Y') + timedelta(days = -365)).strftime('%m/%d/%Y'))
# lstqtr = str((datetime.strptime(td, '%m/%d/%Y') + timedelta(days = -90)).strftime('%m/%d/%Y'))
# tmr = str((datetime.strptime(td, '%m/%d/%Y') + timedelta(days = 1)).strftime('%m/%d/%Y'))

td = str((datetime.today() - timedelta(days = max(1, (datetime.today().weekday() + 6) % 7 - 3))).strftime('%Y-%m-%d'))
tmr = str((datetime.strptime(td, '%Y-%m-%d') + timedelta(days = 1)).strftime('%Y-%m-%d'))
lstyear = str((datetime.strptime(td, '%Y-%m-%d') + timedelta(days = -365)).strftime('%Y-%m-%d'))
lstqtr = str((datetime.strptime(td, '%Y-%m-%d') + timedelta(days = -90)).strftime('%Y-%m-%d'))


initial_fund = pd.read_excel(os.path.join(path,file), sheet_name = cash_tranc)
stock_tranc = pd.read_excel(os.path.join(path,file), sheet_name = holding_tranc)
stock_tranc['total_amt'] = -1 * stock_tranc['amount_per_share'] * stock_tranc['share']

#=========================== Get Cash Value
initial_fund = initial_fund[initial_fund.date<datetime.strptime(td,'%Y-%m-%d')]
stock_tranc = stock_tranc[stock_tranc.date<datetime.strptime(td,'%Y-%m-%d')]

initial_fund_CAD = initial_fund[initial_fund.currency == 'CAD']['value'].sum()
initial_fund_USD = initial_fund[initial_fund.currency == 'USD']['value'].sum()
current_cash_CAD = initial_fund[initial_fund.currency == 'CAD']['value'].sum() + stock_tranc[stock_tranc.currency=='CAD']['total_amt'].agg('sum')
current_cash_USD = initial_fund[initial_fund.currency == 'USD']['value'].sum() + stock_tranc[stock_tranc.currency=='USD']['total_amt'].agg('sum')
current_cash ={
    'currency': ['CAD','USD'],
    'value': [current_cash_CAD, current_cash_USD]
}
current_cash = pd.DataFrame(current_cash)
# curr_cadusd = pd.to_numeric(si.get_data('CADUSD=X', start_date = td, end_date = tmr).sort_index(ascending=False).head(1).adjclose)[0]
curr_cadusd = yf.Ticker("CADUSD=X").history(start = td, end = tmr).sort_index(ascending=True).head(1).Close[0]

#=========================== Get Market Value
current_hold = stock_tranc.groupby('ticker')['share','total_amt'].agg('sum')
current_hold['cost'] = current_hold.total_amt / current_hold.share
current_hold = current_hold.reset_index()
current_hold = current_hold.merge(stock_tranc[['ticker', 'category', 'currency', 'products']], how = 'left', on = 'ticker')

active_stock_ticker = current_hold.ticker.unique()
hold_stock_detail = get_stock_details(active_stock_ticker, lstqtr, td)

from pandasql import sqldf, load_meat, load_births
pysqldf = lambda q: sqldf(q, globals())

stock_curr_price = hold_stock_detail[hold_stock_detail.Close.notnull()].sort_values(by = ['Date'], ascending=False).groupby('ticker').head(3)
stock_curr_price['rank'] = stock_curr_price.groupby('ticker')['Date'].rank(method = 'first', ascending = False)


# In[3]:


# Get Current Holdings
current_hold = stock_tranc.groupby('ticker')['share','total_amt'].agg('sum')
current_hold['cost'] = current_hold.total_amt / current_hold.share
current_hold = current_hold.reset_index()
current_hold = current_hold.merge(stock_tranc[['ticker', 'category', 'currency', 'products']], how = 'left', on = 'ticker')
# print(current_hold)

active_stock_ticker = current_hold.ticker.unique()
hold_stock_detail = get_stock_details(active_stock_ticker, lstqtr, td)

from pandasql import sqldf, load_meat, load_births
pysqldf = lambda q: sqldf(q, globals())

stock_curr_price = hold_stock_detail[hold_stock_detail.Close.notnull()].sort_values(by = ['Date'], ascending=False).groupby('ticker').head(3)
stock_curr_price['rank'] = stock_curr_price.groupby('ticker')['Date'].rank(method = 'first', ascending = False)
query = """
    SELECT 
        A.ticker
        , A.currency
        , A.share
        , A.cost as unit_cost
        , C.Close as lastday_price
        , B.Close as today_price
        , round(A.total_amt) as total_cost
        , round(case when A.ticker = 'CASH' then A.cost else B.Close * A.share end) as curr_value
        , round((B.Close + A.cost) * A.share) as profit
        , round(-1 * B.Close / A.cost - 1, 2) as today_roe
        , round(B.Close / C.Close - 1, 4)  as today_ir
        , A.category
        , A.products
    FROM 
        current_hold A
    LEFT JOIN 
        stock_curr_price B on A.ticker = B.ticker and B.rank = 1
    LEFT JOIN 
        stock_curr_price C on A.ticker = C.ticker and C.rank = 2
        
    UNION
    
    SELECT 
        'CASH' as ticker
        , currency
        , '' as share
        , '' as unit_cost
        , '' as lastday_price
        , '' as today_price
        , '' as total_cost
        , round(value) as curr_value
        , '' as profit
        , '' as today_roe
        , '' as today_ir
        , 'CASH' as category
        , 'CASH' as products
    
    FROM current_cash
    """
stock_curr_smr = pysqldf(query)
stock_curr_smr.insert(11,'value_in_cad', np.where(stock_curr_smr['currency'] == 'CAD', stock_curr_smr['curr_value'], round(stock_curr_smr['curr_value']/curr_cadusd)))
display(stock_curr_smr)


# ## Holdings Summary
# **Portfolio Summary {{ td }}**
# 
# * **Initial Fund**: CAD: {{ locale.currency(initial_fund_CAD, grouping=True) }}, USD: {{ locale.currency(initial_fund_USD, grouping=True)}}, All in CAD: {{ locale.currency(initial_fund_CAD + initial_fund_USD/curr_cadusd, grouping=True)}}.
# * **Current Cash**: CAD: {{locale.currency(current_cash_CAD, grouping=True)}}, USD: {{locale.currency(current_cash_USD, grouping=True)}}, All in CAD: {{locale.currency(current_cash_CAD + current_cash_USD/curr_cadusd, grouping=True)}}
# * **Investment Percentage**: CAD: {{ "{:.1%}".format((stock_curr_smr[stock_curr_smr.currency=='CAD']['curr_value'].sum()-current_cash_CAD)/stock_curr_smr[stock_curr_smr.currency=='CAD']['curr_value'].sum())}}, USD: {{ "{:.1%}".format((stock_curr_smr[stock_curr_smr.currency=='USD']['curr_value'].sum()-current_cash_USD)/stock_curr_smr[stock_curr_smr.currency=='USD']['curr_value'].sum())}}, Overall: {{ "{:.1%}".format((stock_curr_smr[stock_curr_smr.currency=='CAD']['curr_value'].sum() + stock_curr_smr[stock_curr_smr.currency=='USD']['curr_value'].sum()/curr_cadusd - current_cash_CAD - current_cash_USD/curr_cadusd) / (stock_curr_smr[stock_curr_smr.currency=='CAD']['curr_value'].sum() + stock_curr_smr[stock_curr_smr.currency=='USD']['curr_value'].sum()/curr_cadusd))}}
# * **Market Value**: CAD: {{locale.currency(stock_curr_smr[stock_curr_smr.currency=='CAD']['curr_value'].sum(), grouping=True)}}, USD: {{locale.currency(stock_curr_smr[stock_curr_smr.currency=='USD']['curr_value'].sum(), grouping=True)}}, All in CAD: {{locale.currency(stock_curr_smr[stock_curr_smr.currency=='CAD']['curr_value'].sum() + stock_curr_smr[stock_curr_smr.currency=='USD']['curr_value'].sum()/curr_cadusd, grouping=True) }}
# * **Profit**: CAD: {{locale.currency(stock_curr_smr[stock_curr_smr.currency=='CAD']['curr_value'].sum() - initial_fund_CAD, grouping=True)}}, USD: {{locale.currency(stock_curr_smr[stock_curr_smr.currency=='USD']['curr_value'].sum() - initial_fund_USD, grouping=True)}}, All in CAD: {{locale.currency((stock_curr_smr[stock_curr_smr.currency=='CAD']['curr_value'].sum() - initial_fund_CAD) + (stock_curr_smr[stock_curr_smr.currency=='USD']['curr_value'].sum() - initial_fund_USD) /curr_cadusd , grouping=True)}}
# * **ROE**: CAD: {{ "{:.1%}".format((stock_curr_smr[stock_curr_smr.currency=='CAD']['curr_value'].sum() - initial_fund_CAD)/initial_fund_CAD) }}, USD: {{ "{:.1%}".format((stock_curr_smr[stock_curr_smr.currency=='USD']['curr_value'].sum() - initial_fund_USD)/initial_fund_USD) }}, Overall: {{ "{:.1%}".format(((stock_curr_smr[stock_curr_smr.currency=='CAD']['curr_value'].sum() - initial_fund_CAD) + (stock_curr_smr[stock_curr_smr.currency=='USD']['curr_value'].sum() - initial_fund_USD) /curr_cadusd)/(initial_fund_CAD + initial_fund_USD/curr_cadusd)) }}

# ## Fund Distribution

# In[34]:


plot_pie_chart_dist(stock_curr_smr[stock_curr_smr.currency=='USD'], 'products', 'curr_value', 'USD ')
plot_pie_chart_dist(stock_curr_smr[stock_curr_smr.currency=='CAD'], 'products', 'curr_value', 'CAD ')
plot_pie_chart_dist(stock_curr_smr, 'category', 'value_in_cad')


# ## Current Holdings Market Watch

# In[156]:


alt.Chart(hold_stock_detail).mark_line().encode(x='Date',y='Volume', color = 'ticker:N')


# ## Watch List

# In[157]:


watchlist = pd.read_excel(os.path.join(path,file), sheet_name = 'watchlist')
watchlist_df = get_stock_details(watchlist["ticker"].unique(), lstqtr, td)
watchlist_df.head(3)


# In[158]:


alt.Chart(watchlist_df).mark_line().encode(x='Date',y='Volume', color = 'ticker:N')


# In[35]:


get_etf_holding('AWAY', '1mo')


# In[162]:


get_etf_holding('JETS', '1mo')


# In[163]:


get_etf_holding('0P00016N6T.TO', '1mo')


# ## Prepare News Summary

# In[78]:


def get_news_given_ticker(ticker, filter_date):
    news_raw = news.get_yf_rss(ticker)
    n_news = len(news_raw)
    cols = ['title','date','summary','link']
    output = pd.DataFrame(columns=cols)
    for i in range(n_news):
        output.loc[i,'title'] = news_raw[i].title
        output.loc[i,'date'] = timestring.Date(news_raw[i].published).date
        output.loc[i,'summary'] = news_raw[i].summary
        output.loc[i,'link'] = news_raw[i].link
    output = output[output.date >= timestring.Date(filter_date).date].drop(columns = 'date')
    return output

test = get_news_given_ticker('nflx', td)
test

