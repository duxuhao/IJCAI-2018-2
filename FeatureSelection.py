import LRS_SA_RGSS as LSR
from preprocessing import preprocessing
from utility import *
import pandas as pd
import xgboost as xgb
import lightgbm as lgbm
from sklearn.metrics import log_loss

def modelscore(y_test, y_pred):
    return log_loss(y_test, y_pred)

def add(x,y):
    return x + y

def substract(x,y):
    return x - y

def times(x,y):
    return x * y

def divide(x,y):
    return (x + 0.001)/(y + 0.001)

def obtaincol(df, delete):
    ColumnName = list(df.columns)
    for i in delete:
        if i in ColumnName:
            ColumnName.remove(i)
    return ColumnName

def testdata(df,clf,features):
    X = df
    y = df.is_trade
    Loss = []
    T = X.context_timestamp <= '2018-09-23 23:59:47'
    X_train, X_test = X[T], X[~T]
    X_train, X_test = X_train[features], X_test[features]
    y_train, y_test = y[T], y[~T]
    clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=True,early_stopping_rounds=100)
    predict = clf.predict_proba(X_test)[:,1]
    logloss = log_loss(y_test, predict)
    print(logloss)
    return clf

def main(temp, clf, CrossMethod, RecordFolder, test = False):
    df = pd.read_csv('data/train/train.csv')
    df = df[~pd.isnull(df.is_trade)]
#    df1.context_timestamp += 8*60*60
#    df1 = convert_time(df1)
#    features = list(df1.columns)
#    df = pd.read_csv('data/train/round1_ijcai_18_train_20180301.txt',sep=' ')
#    df.context_timestamp += 8*60*60
#    df = preprocessing(df)
#    df = convert_time(df)
    item_category_list_unique = list(np.unique(df.item_category_list))
    df.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)
#    df1.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)
    addcol = []
    '''
       'item_property_list_0', 'item_property_list_1', 'item_property_list_2',
       'item_property_list_3', 'item_property_list_4',
       'predict_category_property_L_0', 'predict_category_property_H_0',
       'predict_category_property_L_1', 'predict_category_property_H_1',
       'predict_category_property_L_2', 'predict_category_property_H_2',
       'predict_category_property_L_3', 'predict_category_property_H_3',
       'predict_category_property_L_4', 'predict_category_property_H_4',]
       'item_id-mean',
       'item_category_list-mean', 'item_brand_id-mean', 'item_city_id-mean',
       'item_price_level-mean', 'user_id-mean', 'user_gender_id-mean',
       'user_occupation_id-mean', 'shop_id-mean', 'shop_review_num_level-mean',
       'shop_score_service-mean', 'shop_score_delivery-mean',
       'shop_score_description-mean', 'hour-mean', 'user_gender_id-hour-mean',
       'user_occupation_id-hour-mean', 'user_age_level-hour-mean',
       'hour-item_category_list-mean', 'hour-item_city_id-mean',
       'hour-item_price_level-mean', 'hour-item_brand_id-mean',
       ]
    '''

    uselessfeatures = ['instance_id', 'item_property_list', 'context_id', 'context_timestamp', 'predict_category_property', 'is_trade']
    ColumnName = obtaincol(df, uselessfeatures) # + addcol #obtain columns withouth the useless features
    print(ColumnName)
    a = LSR.LRS_SA_RGSS_combination(df = df,
                                    clf = clf,
                                    RecordFolder = RecordFolder,
                                    LossFunction = modelscore,
                                    label = 'is_trade',
                                    columnname = ColumnName[::-1],
                                    start = temp,
                                    CrossMethod = CrossMethod,
                                    PotentialAdd = []
                                    )

    try:
        a.run()
    finally:
        with open(RecordFolder, 'a') as f:
            f.write('\n{}\n%{}%\n'.format(type,'-'*60))

if __name__ == "__main__":
    model = {'xgb': xgb.XGBClassifier(seed = 1, max_depth = 5, n_estimators = 2000, nthread = -1),
             'lgb': lgbm.LGBMClassifier(random_state=1,num_leaves = 29, n_estimators=1000),
             'lgb2': lgbm.LGBMClassifier(random_state=1,num_leaves = 29, max_depth=5, n_estimators=1000),
             'lgb3': lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=1000,max_depth=3,learning_rate = 0.09, n_jobs=30),
             'lgb4': lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=5000,max_depth=3,learning_rate = 0.095, n_jobs=30),
             'lgb5': lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=5000,max_depth=3,learning_rate = 0.1, n_jobs=30),
             'lgb6': lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=5000,max_depth=3,learning_rate = 0.05, n_jobs=30)
            }

    CrossMethod = {'+':add,
                   '-':substract,
                   '*':times,
                   '/':divide,}

    RecordFolder = 'record_newselection_lgb6_01.log'
    modelselect = 'lgb6'
    temp = ['item_price_level', 'item_sales_level', 'user_id_query_day', 'shop_score_service', 'hour', 'shop_id_query_day','shop_review_num_level', 'user_age_level', 'user_star_level', 'item_collected_level', 'item_city_id_query_day', 'user_id_query_day_hour', 'user_age_level_query_day', 'item_category_list-mean', 'context_page_id']
    temp = ['item_category_list-mean', 'shop_score_delivery', 'item_sales_level', 'hour', 'item_price_level', 'user_age_level', 'user_star_level',
                'item_collected_level', 'shop_star_level', 'item_pv_level', 'shop_review_positive_rate', 'context_page_id',
                'user_gender_id', 'user_age_level_7', 'item_category_list_7', 'user_occupation_id','user_id_query_day', 'user_id_query_day_hour','user_occupation_id_query_day_hour', 'item_brand_id_query_day',
                  'shop_id_query_day',
                  'shop_score_service',
           ]
    temp = ['item_category_list',
                  'item_city_id',
                  'item_price_level',
                  'item_sales_level',
                  'item_collected_level',
                  'user_gender_id',
                  'user_age_level',
                  'user_star_level',
                  'context_page_id',
                  'shop_review_num_level',
                  'shop_review_positive_rate',
                  'shop_star_level',
                  'shop_score_service',
                  'shop_score_description',
                  'hour',
                  'day',
                  'user_id_query_day',
                  'user_id_query_day_item',
                  'user_id_query_day_hour',
                  'check_ratio',
                  'check_ratio_day_all',
                  'check_time_day',
                  'shop_id',
                  'shop_id_query_day',
#                  'item_brand_id_query_day_hour',
#                  'shop_id_query_day_hour',
#                   'item_city_id_query_day',
#                  'predict_category_property_L_1',
#                  'predict_category_property_H_3', 
#                  'user_occupation_id_query_day_hour', 
#                  'user_id_query_day_hour_item',
                  ]

    temp = [
                  'item_brand_id_query_day_hour', 
                  'user_age_level_query_day', 
                  'user_id_query_day', 
                  'shop_score_description', 
                  'item_city_id_query_day', 
                  'item_category_list', 'item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'hour', 'day', 'user_id_query_day_item', 'user_id_query_day_hour', 'check_ratio', 'check_ratio_day_all', 'check_time_day', 'shop_id', 'item_id_query_day']

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
                  'user_id_query_day_item',
                  'user_id_query_day_hour',
                  'check_ratio',
                  'check_ratio_day_all',
                  'check_time_day',
                  'shop_id',
                  'item_id_query_day',
#                  'user_id_query_day_shop',
#                  'user_id_query_day_hour_shop',
                  'check_shop_ratio',
                  'user_id_query_day_brand_id',
                  'user_id_query_day_hour_brand_id',
                  'check_brand_ratio',
                  'user_id_query_day_item_id',
#                 'check_item_id_ratio',
                 ]
    temp = ['item_brand_id_query_day_hour', 'user_age_level_query_day', 'user_id_query_day', 'shop_score_description', 'item_city_id_query_day', 'item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'hour', 'day', 'user_id_query_day_item', 'user_id_query_day_hour', 'check_ratio', 'check_ratio_day_all', 'check_time_day', 'shop_id', 'item_id_query_day', 'check_item_id_time_day']
    temp = [
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
                  'check_shop_id_ratio',
                  'user_id_query_day_item_brand_id',
                  'user_id_query_day_hour_item_brand_id',
                  'check_item_brand_id_ratio',
                  'user_id_query_day_item_id',
                 ]

    temp = ['item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'hour', 'day', 'user_id_query_day_hour', 'check_item_category_list_ratio', 'check_ratio_day_all', 'check_time_day', 'shop_id', 'item_id_query_day', 'check_shop_id_ratio', 'user_id_query_day_item_brand_id', 'user_id_query_day_hour_item_brand_id', 'check_item_brand_id_ratio', 'user_id_query_day_item_id', 'user_id_query_day', 'item_brand_id']
    temp = ['item_category_list', 'item_price_level', 
                  'item_sales_level', 
                  'item_collected_level', 'item_pv_level', 
                  'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 
                  'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate', 
                  'shop_score_service', 'shop_score_delivery', 'hour', 'day', 'user_id_query_day_hour', 
                  'shop_id', 
                  'item_id_query_day',  'user_id_query_day_item_brand_id', 
                  'user_id_query_day_hour_item_brand_id', 
                  'user_id_query_day', 'item_brand_id','user_id_query_day_item_id', 
                  'check_item_brand_id_ratio', 
                  'check_shop_id_ratio',
                  'check_item_category_list_ratio',
                  'check_ratio_day_all', 
                  'check_time_day',
                  'item_city_id_shop_cnt',
                  'item_city_id_shop_rev_prob',
                  'item_id_shop_rev_cnt',
                  'item_property_list0',
                 ]
    main(temp,model[modelselect], CrossMethod, RecordFolder,test=False)
