# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:13:30 2020

@author: luoyan011
"""



from yahoofinancials import YahooFinancials
commodity_futures = ['GC=F', 'SI=F', 'CL=F']
yahoo_financials_commodities = YahooFinancials(commodity_futures)
daily_commodity_prices = yahoo_financials_commodities.get_historical_price_data('2008-09-15', '2008-09-16', 'daily')


# rows list initialization 
rows = [] 
  
# appending rows 
for data in daily_commodity_prices: 
    for datai in data:
        data_row = datai['currency']
        time = data
    data_row = data['Student'] 
    time = data['Name'] 
      
    for row in data_row: 
        row['Name']= time 
        rows.append(row) 
  
# using data frame 
df = pd.DataFrame(rows) 
  
print(df) 

#https://www.geeksforgeeks.org/python-convert-list-of-nested-dictionary-into-pandas-dataframe/

#https://www.geeksforgeeks.org/python-convert-list-of-nested-dictionary-into-pandas-dataframe/