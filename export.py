import pandas as pd
import numpy as np

if __name__== "__main__":

    data = pd.read_csv('data/train/train2.csv')
    col = ['item_brand_id_user_prob','check_min_difference_ahead','item_category_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'user_gender_id', 'user_age_level', 'user_star_level', 'context_page_id', 'shop_review_positive_rate', 'shop_score_service', 'hour', 'day', 'user_id_query_day_hour', 'shop_id', 'user_id_query_day_item_brand_id', 'user_id_query_day_hour_item_brand_id', 'user_id_query_day', 'user_id_query_day_item_id', 'check_item_brand_id_ratio', 'check_shop_id_ratio', 'check_item_category_list_ratio', 'check_ratio_day_all', 'item_city_id_shop_cnt', 'item_city_id_shop_rev_prob', 'item_id_shop_rev_cnt', 'item_property_list0', 'item_pv_level_price_prob', 'item_collected_level_item_prob', 'item_sales_level_price_prob', 'item_city_id_cnt1d', 'item_collected_level_user_age_cnt', 'item_price_level_item_cnt', 'item_price_level_item_prob', 'user_id_shop_id_trade_meanxd', 'user_age_level_trade_meanxd', 'user_id_item_category_list_tradexd'] # 0078516 008075
    item_category_list_unique = list(np.unique(data.item_category_list))
    data.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)
    train = data[(data['day']) < 24]
    test = data[(data['day'] == 24)] 
    # output train
    X_train = train[col]
    Y_train = train['is_trade']
    subtrain = pd.concat([Y_train, X_train], axis = 1)
    subtrain.to_csv('trainc.csv', header = False, index=False)
    # output test
    X_test = test[col]
    Y_test = test['is_trade']
    subtest = pd.concat([Y_test, X_test], axis = 1)
    subtest.to_csv('testc.csv', header = False, index = False)
