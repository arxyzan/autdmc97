import os
os.system('pip install lightgbm')

import lightgbm as lgb

params = {
  'boosting_type': 'gbrt',
  'objective': 'tweedie',  
  'tweedie_variance_power': 1,
  'metric': {'mape', 'rmse'},
  'learning_rate': 0.03,
  'num_iterations': 10000,
  'num_leaves': 100,
}

n_estimators = 500
