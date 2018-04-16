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
    T = X.context_timestamp <= '2018-09-23 23:59:59'
    X_train, X_test = X[T], X[~T]
    X_train, X_test = X_train[features], X_test[features]
    y_train, y_test = y[T], y[~T]
    clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=True,early_stopping_rounds=100)
    predict = clf.predict_proba(X_test)[:,1]
    logloss = log_loss(y_test, predict)
    print(logloss)
    return clf

def main(temp, clf, CrossMethod, RecordFolder, test = False):
    df = pd.read_csv('data/train/train2.csv')
    df = df[~pd.isnull(df.is_trade)]
    item_category_list_unique = list(np.unique(df.item_category_list))
    df.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)
    uselessfeatures = ['instance_id', 'item_property_list', 'context_id', 'context_timestamp', 'predict_category_property', 'is_trade']
    ColumnName = obtaincol(df, uselessfeatures) # + addcol #obtain columns withouth the useless features
    print(ColumnName)
    a = LSR.LRS_SA_RGSS_combination(df = df,
                                    clf = clf,
                                    RecordFolder = RecordFolder,
                                    LossFunction = modelscore,
                                    label = 'is_trade',
                                    columnname = ColumnName[::3],
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
             'lgb5': lgbm.LGBMClassifier(random_state=1, num_leaves = 13, n_estimators=5000,max_depth=4,learning_rate = 0.05, n_jobs=30),
             'lgb6': lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=5000,max_depth=3,learning_rate = 0.05, n_jobs=8)
            }

    CrossMethod = {'+':add,
                   '-':substract,
                   '*':times,
                   '/':divide,}

    RecordFolder = 'record_fresh_Y_15.log'
    modelselect = 'lgb6'

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
                  'item_pv_level_price_prob',
                  'item_collected_level_item_prob',
                  'item_sales_level_price_prob',
                 ]
    temp = ['item_category_list', 'item_price_level', 'item_sales_level',  'item_pv_level', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'hour', 'day', 'user_id_query_day_hour', 'shop_id', 'item_id_query_day', 'user_id_query_day_item_brand_id', 'user_id_query_day_hour_item_brand_id', 'user_id_query_day', 'user_id_query_day_item_id', 'check_item_brand_id_ratio', 'check_shop_id_ratio', 'check_item_category_list_ratio', 'check_ratio_day_all', 'check_time_day', 'item_city_id_shop_cnt', 'item_city_id_shop_rev_prob', 'item_id_shop_rev_cnt', 'item_property_list0', 'item_pv_level_price_prob', 'item_collected_level_item_prob', 'item_sales_level_price_prob', 'hour_map']
    temp = ['item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'hour', 'day', 'user_id_query_day_hour', 'shop_id', 'item_id_query_day', 'user_id_query_day_item_brand_id', 'user_id_query_day_hour_item_brand_id', 'user_id_query_day', 'item_brand_id', 'user_id_query_day_item_id', 'check_item_brand_id_ratio', 'check_shop_id_ratio', 'check_item_category_list_ratio', 'check_ratio_day_all', 'check_time_day', 'item_city_id_shop_cnt', 'item_city_id_shop_rev_prob', 'item_id_shop_rev_cnt', 'item_property_list0', 'item_pv_level_price_prob', 'item_collected_level_item_prob', 'item_sales_level_price_prob','item_city_id_cnt1d']
    temp = ['item_category_list', 
                  'item_price_level', 
                  'item_sales_level', 
                  'item_collected_level', 
                  #'item_pv_level', 
                  'user_gender_id', 
                  'user_age_level', 
                  'user_occupation_id', 
                  'user_star_level', 
                  'context_page_id', 
                  'shop_review_num_level', 
                  'shop_review_positive_rate', 
                  'shop_score_service', 
                  #'shop_score_delivery', 
                  'hour', 
                  'day', 
                  'user_id_query_day_hour', 
                  'shop_id', 
                  #'item_id_query_day', 
                  'user_id_query_day_item_brand_id', 
                  'user_id_query_day_hour_item_brand_id', 
                  'user_id_query_day',
                  #'item_brand_id',
                  'user_id_query_day_item_id', 
                  'check_item_brand_id_ratio',
                  'check_shop_id_ratio', 
                  'check_item_category_list_ratio', 
                  'check_ratio_day_all', 
                  'check_time_day', 
                  'item_city_id_shop_cnt', 
                  'item_city_id_shop_rev_prob', 
                  'item_id_shop_rev_cnt', 
                  'item_property_list0', 
                  'item_pv_level_price_prob', 
                  'item_collected_level_item_prob',
                  'item_sales_level_price_prob',
                  'item_city_id_cnt1d',
                  'item_collected_level_user_age_cnt',
#                  'user_id_query_day_hour_map_item_pv_level'
                  'check_item_brand_id_time_day'
                 ]
    temp = ['user_occupation_id_tradexd','user_occupation_id_item_price_level_tradexd','user_age_level_shop_review_num_level_tradexd','user_id_shop_id_tradexd','user_id_tradexd','item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery','hour', 'day', 'user_id_query_day_hour', 'shop_id', 'item_id_query_day', 'user_id_query_day_item_brand_id', 'user_id_query_day_hour_item_brand_id', 'user_id_query_day', 'item_brand_id', 'user_id_query_day_item_id', 'check_item_brand_id_ratio', 'check_shop_id_ratio', 'check_item_category_list_ratio', 'check_ratio_day_all', 'check_time_day', 'item_city_id_shop_cnt', 'item_city_id_shop_rev_prob', 'item_id_shop_rev_cnt', 'item_property_list0', 'item_pv_level_price_prob', 'item_collected_level_item_prob', 'item_sales_level_price_prob']
    temp = ['item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_gender_id', 'user_age_level', 'user_star_level', 'context_page_id', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'user_id_query_day_hour', 'shop_id', 'user_id_query_day_item_brand_id', 'user_id_query_day_hour_item_brand_id', 'user_id_query_day', 'user_id_query_day_item_id', 'check_item_brand_id_ratio', 'check_shop_id_ratio', 'check_item_category_list_ratio', 'check_ratio_day_all', 'item_city_id_shop_cnt', 'item_city_id_shop_rev_prob', 'item_id_shop_rev_cnt', 'item_property_list0', 'item_pv_level_price_prob', 'item_collected_level_item_prob', 'item_sales_level_price_prob', 'item_city_id_cnt1d', 'item_collected_level_user_age_cnt', 'item_price_level_item_cnt', 'item_price_level_item_prob', 'user_id_shop_id_trade_meanxd', 'user_age_level_trade_meanxd','user_id_item_category_list_tradexd']
    temp = ['item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 
                 'user_gender_id', 'user_age_level', 'user_star_level', 'context_page_id', 
                 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'user_id_query_day_hour', 
                 'shop_id', 'user_id_query_day_item_brand_id', 'user_id_query_day_hour_item_brand_id', 
                 'user_id_query_day', 'user_id_query_day_item_id', 'check_item_brand_id_ratio', 'check_shop_id_ratio', 
                 'check_item_category_list_ratio', 'check_ratio_day_all', 
                 'item_city_id_shop_cnt', 'item_city_id_shop_rev_prob', 'item_id_shop_rev_cnt', 'item_property_list0', 
                 'item_pv_level_price_prob', 'item_collected_level_item_prob', 'item_sales_level_price_prob', 
                 'item_city_id_cnt1d', 'item_collected_level_user_age_cnt', 
                 #'item_price_level_item_cnt', 'item_price_level_item_prob',
                 'user_id_shop_id_trade_meanxd', 'user_age_level_trade_meanxd',
                 'user_id_item_category_list_tradexd','shop_star_level_user_prob']
    temp = ['item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_gender_id', 'user_age_level', 'user_star_level', 'context_page_id', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'user_id_query_day_hour', 'shop_id', 'user_id_query_day_item_brand_id', 'user_id_query_day_hour_item_brand_id', 'user_id_query_day', 'user_id_query_day_item_id', 'check_item_brand_id_ratio', 'check_shop_id_ratio', 'check_item_category_list_ratio', 'check_ratio_day_all', 'item_city_id_shop_cnt', 'item_city_id_shop_rev_prob', 'item_id_shop_rev_cnt', 'item_property_list0', 'item_pv_level_price_prob', 'item_collected_level_item_prob', 'item_sales_level_price_prob', 'item_city_id_cnt1d', 'item_collected_level_user_age_cnt', 'user_id_shop_id_trade_meanxd', 'user_age_level_trade_meanxd', 'user_id_item_category_list_tradexd', 'shop_star_level_user_prob', 'shop_id_user_prob']
    temp = ['item_price_level_brand_cnt','item_brand_id_user_prob','check_min_difference_ahead','item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_gender_id', 'user_age_level', 'user_star_level', 'context_page_id', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'user_id_query_day_hour', 'shop_id', 'user_id_query_day_item_brand_id', 'user_id_query_day_hour_item_brand_id', 'user_id_query_day', 'user_id_query_day_item_id', 'check_item_brand_id_ratio', 'check_shop_id_ratio', 'check_item_category_list_ratio', 'check_ratio_day_all', 'item_city_id_shop_cnt', 'item_city_id_shop_rev_prob', 'item_id_shop_rev_cnt', 'item_property_list0', 'item_pv_level_price_prob', 'item_collected_level_item_prob', 'item_sales_level_price_prob', 'item_city_id_cnt1d', 'item_collected_level_user_age_cnt', 'item_price_level_item_cnt', 'item_price_level_item_prob', 'user_id_shop_id_trade_meanxd', 'user_age_level_trade_meanxd', 'user_id_item_category_list_tradexd']
    temp = ['item_city_id_item_prob','item_brand_id_shop_rev_prob', 'item_price_level_brand_cnt', 'user_occupation_id_user_age_cnt', 'item_city_id_brand_prob', 'item_brand_id_shop_rev_cnt', 'item_brand_id_user_prob', 'user_id_item_category_list_tradexd','check_min_difference_ahead','item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'user_id_query_day_hour', 'shop_id', 'user_id_query_day_item_brand_id', 'user_id_query_day_hour_item_brand_id', 'user_id_query_day', 'user_id_query_day_item_id', 'check_item_brand_id_ratio', 'check_shop_id_ratio', 'check_item_category_list_ratio', 'check_ratio_day_all', 'check_time_day', 'item_city_id_shop_cnt', 'item_city_id_shop_rev_prob', 'item_id_shop_rev_cnt', 'item_property_list0', 'item_pv_level_price_prob', 'item_collected_level_item_prob', 'item_sales_level_price_prob', 'item_city_id_cnt1d', 'item_collected_level_user_age_cnt', 'item_price_level_item_cnt','item_price_level_item_prob']
    temp = ['item_city_id_item_prob', 'item_brand_id_shop_rev_prob', 'item_price_level_brand_cnt', 'user_occupation_id_user_age_cnt', 'item_city_id_brand_prob', 'item_brand_id_shop_rev_cnt', 'item_brand_id_user_prob', 'user_id_item_category_list_tradexd', 'check_min_difference_ahead', 'item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'user_id_query_day_hour', 'shop_id', 'user_id_query_day_item_brand_id', 'user_id_query_day_hour_item_brand_id', 'user_id_query_day', 'user_id_query_day_item_id', 'check_item_brand_id_ratio', 'check_shop_id_ratio', 'check_item_category_list_ratio', 'check_ratio_day_all', 'check_time_day', 'item_city_id_shop_cnt', 'item_city_id_shop_rev_prob', 'item_id_shop_rev_cnt', 'item_property_list0', 'item_pv_level_price_prob', 'item_collected_level_item_prob', 'item_sales_level_price_prob', 'item_city_id_cnt1d', 'item_collected_level_user_age_cnt', 'item_price_level_item_cnt', 'item_price_level_item_prob', 'user_id_shop_prob', 'user_id_item_cnt']
    temp = ['item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_gender_id', 'user_age_level', 'user_star_level', 'context_page_id', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day']
    temp = ['user_id_query_day','user_id_query_day_item_id','check_ratio_day_all','item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_gender_id', 'user_age_level', 'user_star_level', 'context_page_id', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'check_min_difference_ahead', 'check_item_brand_id_ratio', 'user_id_query_min_map']
    temp = ['user_id_query_day_item_id', 'check_ratio_day_all', 'item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_gender_id', 'user_age_level', 'user_star_level', 'context_page_id', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'check_min_difference_ahead', 'check_item_brand_id_ratio', 'user_id_query_min_map', 'item_brand_id_shop_cnt', 'user_id_query_day']
    temp = ['user_id_query_day_item_id', 'check_ratio_day_all', 'item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_gender_id', 'user_age_level', 'user_star_level', 'context_page_id', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'check_min_difference_ahead', 'check_item_brand_id_ratio', 'user_id_query_min_map', 'item_brand_id_shop_cnt', 'user_id_query_day', 'user_gender_id_item_id_trade_meanxd']
    temp = ['user_id_shop_cnt','user_id_query_day_item_id', 'check_ratio_day_all', 'item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_gender_id', 'user_age_level', 'user_star_level', 'context_page_id', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'check_min_difference_ahead', 'check_item_brand_id_ratio', 'user_id_query_min_map', 'item_brand_id_shop_cnt', 'user_id_query_day', 'user_gender_id_item_id_trade_meanxd', 'user_id_query_day_hour_item_pv_level']
    temp = ['user_id_shop_cnt', 'user_id_query_day_item_id', 'check_ratio_day_all', 'item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_age_level', 'user_star_level', 'context_page_id', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'check_min_difference_ahead', 'check_item_brand_id_ratio', 'user_id_query_min_map', 'item_brand_id_shop_cnt', 'user_id_query_day', 'user_gender_id_item_id_trade_meanxd', 'item_sales_level_city_cnt', 'user_id_item_category_list_trade_meanxd']
    temp = ['user_gender_id_item_city_id_tradexd', 'user_id_shop_cnt', 'user_id_query_day_item_id', 'check_ratio_day_all', 'item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_age_level', 'user_star_level', 'context_page_id', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'check_min_difference_ahead', 'check_item_brand_id_ratio', 'user_id_query_min_map', 'item_brand_id_shop_cnt', 'user_id_query_day', 'user_gender_id_item_id_trade_meanxd', 'item_sales_level_city_cnt', 'user_id_item_category_list_trade_meanxd', 'predict_category_property0']
    temp = ['user_id_shop_cnt', 'user_id_query_day_item_id', 'check_ratio_day_all', 'item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_age_level', 'user_star_level', 'context_page_id', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'check_min_difference_ahead', 'check_item_brand_id_ratio', 'user_id_query_min_map', 'item_brand_id_shop_cnt', 'user_id_query_day', 'user_gender_id_item_id_trade_meanxd', 'item_sales_level_city_cnt', 'user_id_item_category_list_trade_meanxd', 'predict_category_property0', 'user_gender_id_item_city_id_tradexd']
    temp = ['user_id_shop_cnt', 'user_id_query_day_item_id', 'check_ratio_day_all', 'item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_age_level', 'user_star_level', 'context_page_id', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'check_min_difference_ahead', 'check_item_brand_id_ratio', 'user_id_query_min_map', 'item_brand_id_shop_cnt', 'user_id_query_day', 'user_gender_id_item_id_trade_meanxd', 'item_sales_level_city_cnt', 'user_id_item_category_list_trade_meanxd', 'predict_category_property0', 'user_gender_id_item_city_id_tradexd', 'user_occupation_id_user_cnt', 'check_item_sales_level_ratio_hour_map']
    temp = ['user_id_shop_cnt', 'user_id_query_day_item_id', 'check_ratio_day_all', 'item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_age_level', 'user_star_level', 'context_page_id', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'check_min_difference_ahead', 'check_item_brand_id_ratio', 'user_id_query_min_map', 'item_brand_id_shop_cnt', 'user_id_query_day', 'user_gender_id_item_id_trade_meanxd', 'item_sales_level_city_cnt', 'user_id_item_category_list_trade_meanxd', 'predict_category_property0', 'user_gender_id_item_city_id_tradexd', 'user_occupation_id_user_cnt', 'check_item_sales_level_ratio_hour_map', 'user_id_coll_prob']
    temp = ['check_ratio_day_all','check_min_difference_ahead']
    temp = ['check_ratio_day_all', 'check_min_difference_ahead', 'user_occupation_id_item_id_trade_meanxd', 'sale_collect', 'user_id_query_day_item_id', 'item_price_level_user_age_cnt', 'hour', 'item_id_user_prob', 'user_id_tradexd', 'check_item_brand_id_ratio', 'item_price_level_shop_cnt', 'item_city_id_query_day', 'item_id_user_occ_prob']
    temp = ['check_ratio_day_all', 'check_min_difference_ahead', 'user_occupation_id_item_id_trade_meanxd', 'sale_collect', 'user_id_query_day_item_id', 'item_price_level_user_age_cnt', 'hour', 'item_id_user_prob', 'user_id_tradexd', 'check_item_brand_id_ratio', 'item_price_level_shop_cnt', 'item_city_id_query_day', 'item_id_user_occ_prob', 'user_star_level_user_gender_cnt', 'item_category_list2', 'user_id_shop_id_tradexd']
    main(temp,model[modelselect], CrossMethod, RecordFolder,test=False)
