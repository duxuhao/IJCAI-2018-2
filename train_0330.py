from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from multiprocessing import Pool
import lightgbm as lgbm
import xgboost as xgb
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from utility import convert_time

def run(features, label, df, clf):
    print(features[-1])
    X = df
    y = df[label]
    Loss = []
    T = X.context_timestamp <= '2018-09-23 23:59:59'
    X_train, X_test = X[T], X[~T]
    X_train, X_test = X_train[features], X_test[features]
    #norm = StandardScaler()
    #X_train = norm.fit_transform(X_train[features])
    #X_test = norm.transform(X_test[features])
    y_train, y_test = y[T], y[~T]
    clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=True,early_stopping_rounds=200)
    predict = clf.predict_proba(X_test)[:,1]
    logloss = log_loss(y_test, predict)
    print(logloss)
    return clf

def Myprediction(df, features, clf, name, item_category_list_unique):
    testdf = pd.read_csv('data/test/round1_ijcai_18_test_a_20180301.txt',sep=' ')
    testdf.context_timestamp += 8*60*60
    testdf = convert_time(testdf)
    testdf.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)
    prediction_format = pd.read_csv('data/output/0203.txt',sep=' ')
    train, predict = df, testdf
    clf.fit(train[features], train.is_trade, eval_set = [(train[features], train.is_trade)], eval_metric='logloss', verbose=True)
    predict['predicted_score'] = clf.predict_proba(predict[features])[:,1]
    print(predict[['instance_id', 'predicted_score']])
    prediction_file = pd.merge(prediction_format[['instance_id']], predict[['instance_id', 'predicted_score']], on = 'instance_id', how = 'left')
    prediction_file.to_csv('data/output/{}.txt'.format(name), sep=' ',index = None)
    return clf

df = pd.read_csv('data/train/round1_ijcai_18_train_20180301.txt',sep=' ')
df.context_timestamp += 8*60*60
df = convert_time(df)
item_category_list_unique = list(np.unique(df.item_category_list))
df.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)
label = 'is_trade'

start_features = [
                  'item_category_list',
                  'item_city_id',
                  'item_price_level',
                  'item_sales_level',
                  'item_collected_level',
                  'item_pv_level',
                  'user_gender_id',
                  'user_age_level',
                  'user_occupation_id',
                  'user_star_level',
                  'context_page_id',
                  'shop_review_num_level',
                  'shop_review_positive_rate',
#                  'shop_star_level',
                  'shop_score_service',
                  'shop_score_delivery',
                  'hour',
                  'day',
                  'user_id_query_day_item_category_list',
                  'user_id_query_day_hour',
                  'check_item_category_list_ratio',
                  'check_ratio_day_all',
                  'check_time_day',
                  'shop_id',
                  'item_id_query_day',

#                  'user_id_query_day_shop',
#                  'user_id_query_day_hour_shop',
                  'check_shop_id_ratio',
                  'user_id_query_day_item_brand_id',
                  'user_id_query_day_hour_item_brand_id',
                  'check_item_brand_id_ratio',
                  'user_id_query_day_item_id',
                 ]

clf = lgbm.LGBMClassifier(random_state=1, num_leaves = 6,
                          n_estimators=739,max_depth=3,learning_rate = 0.08, n_jobs=30) #008154
#clf =  xgb.XGBClassifier(seed = 1, max_depth=3, n_estimators=1000)
clf = run(start_features, label, df, clf)
#Myprediction(df, start_features, clf, 'Peter_0330', item_category_list_unique)
