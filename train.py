import pandas as pd
import numpy as np
import lightgbm as lgb
from utility import convert_time

def train_clf(train, test, col, clf):
    X = train[col]
    y = train['is_trade']
    X_tes = test[col]
    y_tes = test['is_trade']
    print(X.shape)
    clf.fit(X, y, eval_set=[(X_tes, y_tes)], early_stopping_rounds=200)
    iteration = clf.best_iteration_
    pred = clf.predict_proba(test[col])[:, 1]
    return iteration

def prediction(train, test, col, clf):
    X = train[col]
    y = train['is_trade']
    clf.fit(X, y, eval_set=[(X, y)])
    pred = clf.predict_proba(test[col])[:, 1]
    test['predicted_score'] = pred
    sub1 = test[['instance_id', 'predicted_score']]
    sub=pd.read_csv('data/test/round1_ijcai_18_test_a_20180301.txt',sep=' ')
    sub=pd.merge(sub,sub1,on=['instance_id'],how='left')
    sub=sub.fillna(0)
    sub[['instance_id', 'predicted_score']].to_csv('data/output/robert_0404.txt',sep=" ",index=False)


if __name__ == "__main__":
    data = pd.read_csv('data/train/train2.csv')
    item_category_list_unique = list(np.unique(data.item_category_list))
    data.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)
    train= data[(data['day'] < 24)]
    test= data[(data['day'] == 24)]
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
    col = ['item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'hour', 'day', 'user_id_query_day_hour', 'shop_id', 'item_id_query_day', 'user_id_query_day_item_brand_id', 'user_id_query_day_hour_item_brand_id', 'user_id_query_day', 'user_id_query_day_item_id', 'check_item_brand_id_ratio', 'check_shop_id_ratio', 'check_item_category_list_ratio', 'check_ratio_day_all', 'check_time_day', 'item_city_id_shop_cnt', 'item_city_id_shop_rev_prob', 'item_id_shop_rev_cnt', 'item_property_list0', 'item_pv_level_price_prob', 'item_collected_level_item_prob', 'item_sales_level_price_prob', 'hour_map'] #Y 0.07887 008103
    col = ['item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'hour', 'day', 'user_id_query_day_hour', 'shop_id', 'item_id_query_day', 'user_id_query_day_item_brand_id', 'user_id_query_day_hour_item_brand_id', 'user_id_query_day', 'item_brand_id', 'user_id_query_day_item_id', 'check_item_brand_id_ratio', 'check_shop_id_ratio', 'check_item_category_list_ratio', 'check_ratio_day_all', 'check_time_day', 'item_city_id_shop_cnt', 'item_city_id_shop_rev_prob', 'item_id_shop_rev_cnt', 'item_property_list0', 'item_pv_level_price_prob', 'item_collected_level_item_prob', 'item_sales_level_price_prob'] # robert 007894 008100
    col = ['item_category_list', 
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
                 ] # 0.0787838429897982 008099
    col = ['item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'user_id_query_day_hour', 'shop_id', 'user_id_query_day_item_brand_id', 'user_id_query_day_hour_item_brand_id', 'user_id_query_day', 'user_id_query_day_item_id', 'check_item_brand_id_ratio', 'check_shop_id_ratio', 'check_item_category_list_ratio', 'check_ratio_day_all', 'check_time_day', 'item_city_id_shop_cnt', 'item_city_id_shop_rev_prob', 'item_id_shop_rev_cnt', 'item_property_list0', 'item_pv_level_price_prob', 'item_collected_level_item_prob', 'item_sales_level_price_prob', 'item_city_id_cnt1d', 'item_collected_level_user_age_cnt', 'user_id_query_day_hour_map_item_pv_level'] # 0.078648
    clf = lgb.LGBMClassifier(random_state=1, num_leaves = 6,
                          n_estimators=200000,max_depth=3,learning_rate = 0.05,
                          subsample=1, n_jobs=30)
    iteration = train_clf(train, test, col, clf)
#    iteration = 1032
    clf.n_estimators = iteration
    train = data[~pd.isnull(data.is_trade)]
    test = data[pd.isnull(data.is_trade)]
    prediction(train, test, col, clf)
