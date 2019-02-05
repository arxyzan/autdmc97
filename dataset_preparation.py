import pandas as pd

#read from csv
dataset = pd.read_csv('data.csv')

#sort by date for visualization
dataset = dataset[['Log_Date', 'FROM', 'TO']].sort_values('Log_Date')

#calculat sales values
dataset = dataset.groupby(['Log_Date', 'FROM', 'TO']).size()
dataset = dataset.reset_index(level=['Log_Date', 'FROM', 'TO'])
dataset = dataset.rename(columns={0: 'SALES'})

#split log date by year, month and day
dataset['YEAR'] = dataset.Log_Date.str.split('/').str[0]
dataset['MONTH'] = dataset.Log_Date.str.split('/').str[1]
dataset['DAY'] = dataset.Log_Date.str.split('/').str[2]

#convert all values to type int64
dataset = dataset[['YEAR', 'MONTH', 'DAY', 'FROM', 'TO', 'SALES']].astype('int64')
dataset = dataset.dropna()
dataset.isna().sum()


#save to csv
dataset.to_csv('dataset_with_sales.csv')









