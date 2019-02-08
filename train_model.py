from dataset_preparation import *
from lgbm import *
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#read data from dataset
data = dataset[["YEAR", "MONTH", "DAY", "FROM", "TO","SALES"]]


#create train data and test data

train_dataset = data.sample(frac=0.9,random_state=0)
test_dataset = data.drop(train_dataset.index)

#setting labels

train_labels = train_dataset.pop('SALES')
test_labels = test_dataset.pop('SALES')

y_train = train_labels
y_test = test_labels
X_train = train_dataset
X_test = test_dataset


lgb_train = lgb.Dataset(X_train, y_train, 
                        feature_name=["YEAR", "MONTH", "DAY", "FROM", "TO"], 
                        categorical_feature=['MONTH', 'DAY', 'FROM', 'TO'])
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

gbm = lgb.train(params,
                lgb_train,
                n_estimators,
                valid_sets=lgb_eval,
                early_stopping_rounds=500)


#predict on eval data
print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=best_iteration)
# eval
print('MAPE:', mape(y_test, y_pred))
print('MAE:', mean_squared_error(y_test, y_pred)**0.5)


#plot results

plt.figure(figsize = (20, 12))
plt.plot(y_pred[:100])
plt.plot(test_labels.values[:100])

#applying prediction on test data

test_dataset = pd.read_csv("test.csv") 
test_dataset = test_dataset[[["YEAR", "MONTH", "DAY", "FROM", "TO"]]]

test_dataset['YEAR'] = test_dataset.Log_Date.str.split('/').str[0]
test_dataset['MONTH'] = test_dataset.Log_Date.str.split('/').str[1]
test_dataset['DAY'] = test_dataset.Log_Date.str.split('/').str[2]

# predict
print('Starting prediction...')
test_pred = gbm.predict(test_dataset, num_iteration=gbm.best_iteration)
print("done!")

for i in range(0,len(test_pred)):
  test_pred[i] = round(test_pred[i])

result_test = pd.read_csv("test.csv") 
result_test['SALES'] = test_pred

result_test.to_csv('result.csv')



