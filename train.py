import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn import preprocessing
import warnings
from utility import convert_time

warnings.filterwarnings("ignore")

def lgbCV(train, test, col):
    X = train[col]
    y = train['is_trade'].values
    X_tes = test[col]
    y_tes = test['is_trade'].values
    print(X.shape)
    print('Training LGBM model...')
    lgb0 = lgb.LGBMClassifier(
        objective='binary',
        # metric='binary_error',
#        num_leaves=35,
#        max_depth=8,
        num_leaves=8,
        max_depth=3,
        learning_rate=0.05,
        seed=2018,
#        colsample_bytree=0.8,
        # min_child_samples=8,
#        subsample=0.9,
        n_estimators=20000,
        n_jobs=30)
    lgb0 = lgb.LGBMClassifier(random_state=1, num_leaves = 6,
                          n_estimators=5000,max_depth=3,learning_rate = 0.05, 
                          subsample=1, n_jobs=30)
    lgb_model = lgb0.fit(X, y, eval_set=[(X_tes, y_tes)], early_stopping_rounds=200)
    best_iter = lgb_model.best_iteration_
    #predictors = [i for i in X.columns]
    #feat_imp = pd.Series(lgb_model.feature_importance(), predictors).sort_values(ascending=False)
    #print(feat_imp)
    #print(feat_imp.shape)
    # pred= lgb_model.predict(test[col])
    pred = lgb_model.predict_proba(test[col])[:, 1]
    test['pred'] = pred
    test['index'] = range(len(test))
    # print(test[['is_trade','pred']])
    print('误差 ', log_loss(test['is_trade'], test['pred']))
    return best_iter

def sub(train, test, col, best_iter):
    X = train[col]
    y = train['is_trade'].values
    print('Training LGBM model...')
    lgb0 = lgb.LGBMClassifier(
        objective='binary',
        # metric='binary_error',
        num_leaves=8,
        max_depth=3,
        learning_rate=0.05,
        seed=2018,
#        colsample_bytree=0.8,
        # min_child_samples=8,
#        subsample=0.9,
        n_estimators=best_iter,
        n_jobs=30)
    lgb0  = lgb.LGBMClassifier(random_state=1, num_leaves = 6,
                          n_estimators=best_iter,max_depth=3,learning_rate = 0.05, 
                          subsample=1, n_jobs=30)
    lgb_model = lgb0.fit(X, y, eval_set=[(X, y)])
    #predictors = [i for i in X.columns]
    #feat_imp = pd.Series(lgb_model.feature_importance(), predictors).sort_values(ascending=False)
    #print(feat_imp)
    #print(feat_imp.shape)
    # pred= lgb_model.predict(test[col])
    pred = lgb_model.predict_proba(test[col])[:, 1]
    test['predicted_score'] = pred
    sub1 = test[['instance_id', 'predicted_score']]
    sub=pd.read_csv('data/test/round1_ijcai_18_test_a_20180301.txt',sep=' ')
    sub=pd.merge(sub,sub1,on=['instance_id'],how='left')
    sub=sub.fillna(0)
    #sub[['instance_id', 'predicted_score']].to_csv('result/result0320.csv',index=None,sep=' ')
    sub[['instance_id', 'predicted_score']].to_csv('data/output/Peter_0402.txt',sep=" ",index=False)


if __name__ == "__main__":
    data = pd.read_csv('data/train/train.csv')
    item_category_list_unique = list(np.unique(data.item_category_list))
    data.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)
    train= data[(data['day'] >= 18) & (data['day'] <= 23)]
    test= data[(data['day'] == 24)]
    col = [c for c in train if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp']]

    for i in col[:]:
        if ('check' in i) | ('query' in i):
            col.remove(i)

    col = col + ['item_category_list', 'user_id_query_day_hour', 'shop_id','item_id_query_day',  'user_id_query_day_item_brand_id', 'user_id_query_day_hour_item_brand_id', 'user_id_query_day', 'item_brand_id','user_id_query_day_item_id', 'check_item_brand_id_ratio', 'check_shop_id_ratio','check_item_category_list_ratio','check_ratio_day_all','check_time_day']
    col = ['item_category_list', 
                  'item_price_level', 
                  'item_sales_level', 
                  'item_collected_level', 'item_pv_level', 
                  'user_gender_id', 'user_age_level', 'user_occupation_id', 
                  'user_star_level', 
                  'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate', 
                  'shop_score_service', 'shop_score_delivery', 'hour', 'day', 
                  'user_id_query_day_hour', 
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
                 ] # 008104
    col = ['item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'hour', 'day', 'user_id_query_day_hour', 'shop_id', 'item_id_query_day', 'user_id_query_day_item_brand_id', 'user_id_query_day_hour_item_brand_id', 'user_id_query_day', 'item_brand_id', 'user_id_query_day_item_id', 'check_item_brand_id_ratio', 'check_shop_id_ratio', 'check_item_category_list_ratio', 'check_ratio_day_all', 'check_time_day', 'item_city_id_shop_cnt', 'item_city_id_shop_rev_prob', 'item_id_shop_rev_cnt', 'item_property_list0', 'item_pv_level_price_prob', 'item_pv_level_shop_rev_cnt', 'item_city_id_shop_prob']

    best_iter = lgbCV(train, test, col)
    "----------------------------------------------------线上----------------------------------------"
    train = data[data.is_trade.notnull()]
    test = data[data.is_trade.isnull()]
    sub(train, test, col, best_iter)
