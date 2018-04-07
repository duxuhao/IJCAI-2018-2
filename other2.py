import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

data_root = 'data'

def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


def convert_data(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    print(set(data.day))

    data['item_category_list_1'] = data['item_category_list'].apply(lambda x: int(x.split(';')[0]))
    data['item_category_list_2'] = data['item_category_list'].apply(lambda x: int(x.split(';')[1]))
    data['item_property_list_0'] = data['item_property_list'].apply(lambda x: int(x.split(';')[0]))
    data['item_property_list_1'] = data['item_property_list'].apply(lambda x: int(x.split(';')[1]))
    data['item_property_list_2'] = data['item_property_list'].apply(lambda x: int(x.split(';')[2]))
    for i in range(3):
        data['predict_category_%d' % (i)] = data['predict_category_property'].apply(

            lambda x: int(str(x.split(";")[i]).split(":")[0]) if len(x.split(";")) > i else -1

        )

    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left', on=['user_id', 'day', 'hour'])

    user_item_day = data.groupby(['user_id', 'day', 'item_id']).size().reset_index().rename(columns={0: 'user_item_query_day'})
    data = pd.merge(data, user_item_day, 'left', on=['user_id', 'day', 'item_id'])
    user_item_day_hour = data.groupby(['user_id', 'day', 'hour', 'item_id']).size().reset_index().rename(
        columns={0: 'user_item_day_hour'})
    data = pd.merge(data, user_item_day_hour, 'left', on=['user_id', 'day', 'hour', 'item_id'])

    user_item_brand_day = data.groupby(['user_id', 'day', 'item_brand_id']).size().reset_index().rename(
        columns={0: 'user_item_brand_query_day'})
    data = pd.merge(data, user_item_brand_day, 'left', on=['user_id', 'day', 'item_brand_id'])
    user_item_brand_day_hour = data.groupby(['user_id', 'day', 'hour', 'item_brand_id']).size().reset_index().rename(
        columns={0: 'user_item_brand_day_hour'})
    data = pd.merge(data, user_item_brand_day_hour, 'left', on=['user_id', 'day', 'hour', 'item_brand_id'])

    user_item_sales_level_day = data.groupby(['user_id', 'day'], as_index=False)['item_sales_level']\
        .agg({'user_item_sales_level_day_mean': 'mean',
              'user_item_sales_level_day_median': 'median',
              'user_item_sales_level_day_min': 'min',
              'user_item_sales_level_day_max': 'max',
              'user_item_sales_level_day_std': 'std',
              'user_item_sales_level_day_count': 'count'})
    data = pd.merge(data, user_item_sales_level_day, 'left', on=['user_id', 'day'])
    user_item_sales_level_day_hour = data.groupby(['user_id', 'day', 'hour'], as_index=False)['item_sales_level']\
        .agg({'user_item_sales_level_hour_mean': 'mean',
              'user_item_sales_level_hour_median': 'median',
              'user_item_sales_level_hour_min': 'min',
              'user_item_sales_level_hour_max': 'max',
              'user_item_sales_level_hour_std': 'std',
              'user_item_sales_level_hour_count': 'count'})
    data = pd.merge(data, user_item_sales_level_day_hour, 'left', on=['user_id', 'day', 'hour'])

    user_item_collected_level_day = data.groupby(['user_id', 'day'], as_index=False)['item_collected_level'] \
        .agg({'user_item_collected_level_day_mean': 'mean',
              'user_item_collected_level_day_median': 'median',
              'user_item_collected_level_day_min': 'min',
              'user_item_collected_level_day_max': 'max',
              'user_item_collected_level_day_std': 'std',
              'user_item_collected_level_day_count': 'count'})
    data = pd.merge(data, user_item_collected_level_day, 'left', on=['user_id', 'day'])
    user_item_collected_level_day_hour = data.groupby(['user_id', 'day', 'hour'], as_index=False)['item_collected_level'] \
        .agg({'user_item_collected_level_hour_mean': 'mean',
              'user_item_collected_level_hour_median': 'median',
              'user_item_collected_level_hour_min': 'min',
              'user_item_collected_level_hour_max': 'max',
              'user_item_collected_level_hour_std': 'std',
              'user_item_collected_level_hour_count': 'count'})
    data = pd.merge(data, user_item_collected_level_day_hour, 'left', on=['user_id', 'day', 'hour'])

    user_occupation_id_query_day = data.groupby(['user_occupation_id', 'day']).size().reset_index().rename(columns={0: 'user_occupation_id_query_day'})
    data = pd.merge(data, user_occupation_id_query_day, 'left', on=['user_occupation_id', 'day'])
    user_occupation_id_query_day_hour = data.groupby(['user_occupation_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_occupation_id_query_day_hour'})
    data = pd.merge(data, user_occupation_id_query_day_hour, 'left', on=['user_occupation_id', 'day', 'hour'])

    user_star_level_query_day = data.groupby(['user_star_level', 'day']).size().reset_index().rename(
        columns={0: 'user_star_level_query_day'})
    data = pd.merge(data, user_star_level_query_day, 'left', on=['user_star_level', 'day'])
    user_star_level_query_day_hour = data.groupby(['user_star_level', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_star_level_query_day_hour'})
    data = pd.merge(data, user_star_level_query_day_hour, 'left', on=['user_star_level', 'day', 'hour'])

    item_category_list_1_day = data.groupby(['item_category_list_1', 'day']).size().reset_index().rename(
        columns={0: 'item_category_list_1_query_day'})
    data = pd.merge(data, item_category_list_1_day, 'left', on=['item_category_list_1', 'day'])
    item_category_list_1_hour = data.groupby(['item_category_list_1', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'item_category_list_1_query_day_hour'})
    data = pd.merge(data, item_category_list_1_hour, 'left', on=['item_category_list_1', 'day', 'hour'])

    item_brand_id_day = data.groupby(['item_brand_id', 'day']).size().reset_index().rename(
        columns={0: 'item_brand_id_day'})
    data = pd.merge(data, item_brand_id_day, 'left', on=['item_brand_id', 'day'])
    item_brand_id_hour = data.groupby(['item_brand_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'item_brand_id_hour'})
    data = pd.merge(data, item_brand_id_hour, 'left', on=['item_brand_id', 'day', 'hour'])

    item_pv_level_day = data.groupby(['item_pv_level', 'day']).size().reset_index().rename(
        columns={0: 'item_pv_level_day'})
    data = pd.merge(data, item_pv_level_day, 'left', on=['item_pv_level', 'day'])
    item_pv_level_hour = data.groupby(['item_pv_level', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'item_pv_level_hour'})
    data = pd.merge(data, item_pv_level_hour, 'left', on=['item_pv_level', 'day', 'hour'])

    shop_star_level_day = data.groupby(['shop_star_level', 'day']).size().reset_index().rename(
        columns={0: 'shop_star_level_day'})
    data = pd.merge(data, shop_star_level_day, 'left', on=['shop_star_level', 'day'])
    shop_star_level_hour = data.groupby(['shop_star_level', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'shop_star_level_hour'})
    data = pd.merge(data, shop_star_level_hour, 'left', on=['shop_star_level', 'day', 'hour'])

    shop_score_service_day = data.groupby(['day'], as_index=False)['shop_score_service'] \
        .agg({'shop_score_service_day_mean': 'mean',
              'shop_score_service_day_median': 'median',
              'shop_score_service_day_min': 'min',
              'shop_score_service_day_max': 'max',
              'shop_score_service_day_std': 'std',
              'shop_score_service_day_count': 'count'})
    data = pd.merge(data, shop_score_service_day, 'left', on=['day'])
    shop_score_service_hour = data.groupby(['day', 'hour'], as_index=False)[
        'shop_score_service'] \
        .agg({'shop_score_service_hour_mean': 'mean',
              'shop_score_service_hour_median': 'median',
              'shop_score_service_hour_min': 'min',
              'shop_score_service_hour_max': 'max',
              'shop_score_service_hour_std': 'std',
              'shop_score_service_hour_count': 'count'})
    data = pd.merge(data, shop_score_service_hour, 'left', on=['day', 'hour'])

    user_occupation_id_day = data.groupby(['day'], as_index=False)['user_occupation_id'] \
        .agg({'user_occupation_id_day_mean': 'mean',
              'user_occupation_id_day_median': 'median',
              'user_occupation_id_day_min': 'min',
              'user_occupation_id_day_max': 'max',
              'user_occupation_id_day_std': 'std',
              'user_occupation_id_day_count': 'count'})
    data = pd.merge(data, user_occupation_id_day, 'left', on=['day'])
    user_occupation_id_hour = data.groupby(['day', 'hour'], as_index=False)[
        'user_occupation_id'] \
        .agg({'user_occupation_id_hour_mean': 'mean',
              'user_occupation_id_hour_median': 'median',
              'user_occupation_id_hour_min': 'min',
              'user_occupation_id_hour_max': 'max',
              'user_occupation_id_hour_std': 'std',
              'user_occupation_id_hour_count': 'count'})
    data = pd.merge(data, user_occupation_id_hour, 'left', on=['day', 'hour'])

    item_pv_level_day = data.groupby(['day'], as_index=False)['item_pv_level'] \
        .agg({'item_pv_level_day_mean': 'mean',
              'item_pv_level_day_median': 'median',
              'item_pv_level_day_min': 'min',
              'item_pv_level_day_max': 'max',
              'item_pv_level_day_std': 'std',
              'item_pv_level_day_count': 'count'})
    data = pd.merge(data, item_pv_level_day, 'left', on=['day'])
    item_pv_level_hour = data.groupby(['day', 'hour'], as_index=False)[
        'predict_category_0'] \
        .agg({'item_pv_level_hour_mean': 'mean',
              'item_pv_level_hour_median': 'median',
              'item_pv_level_hour_min': 'min',
              'item_pv_level_hour_max': 'max',
              'item_pv_level_hour_std': 'std',
              'item_pv_level_hour_count': 'count'})
    data = pd.merge(data, item_pv_level_hour, 'left', on=['day', 'hour'])

    return data


if __name__ == "__main__":
    online = False # 这里用来标记是 线下验证 还是 在线提交

    data = pd.read_csv('{}/round1_ijcai_18_train_20180301.txt'.format(data_root), sep=' ')
    test = pd.read_csv('{}/round1_ijcai_18_test_a_20180301.txt'.format(data_root), sep=' ')
    ntest = test.shape[0]
    ntrain = data.shape[0]
    data.drop_duplicates(inplace=True)
    data = convert_data(data)
    test = convert_data(test)

    features = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                'item_pv_level', 'item_pv_level',
                'user_gender_id', 'user_occupation_id',
                'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
                'context_page_id', 'hour', 'shop_id', 'shop_review_num_level', 'shop_star_level',
                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
                'item_property_list_0', 'item_property_list_2', 'item_category_list_1', 'item_category_list_2',
                'predict_category_0', 'predict_category_1',
                'user_item_query_day', 'user_item_day_hour', 'user_item_brand_query_day', 'user_item_brand_day_hour',
                'user_item_sales_level_hour_mean', 'user_item_sales_level_hour_median', 'user_item_sales_level_hour_min',
                'user_item_sales_level_hour_max', 'user_item_sales_level_hour_count',
                'user_item_collected_level_hour_mean', 'user_item_collected_level_hour_median', 'user_item_collected_level_hour_std',
                'user_occupation_id_query_day', 'user_occupation_id_query_day_hour',
                'item_category_list_1_query_day', 'item_category_list_1_query_day_hour',
                'item_brand_id_day', 'item_brand_id_hour', 'shop_star_level_day',
                'shop_score_service_day_count', 'shop_score_service_hour_count',
                'all_query_day', 'all_query_hour'
                ]
    target = ['is_trade']

    params = {
        'boosting': 'gbdt',
        'application': 'binary',
        'metric': 'binary_logloss',
        'bagging_fraction': 0.80,
        'feature_fraction': 0.80,
        'num_leaves': 32,
        'learning_rate': 0.1,
        'max_depth': 5,
        'max_bin': 255,
        'min_data_in_leaf': 100,
        'lambda_l1': 3.0,
        'lambda_l2': 3.0,
        'num_threads': 10,
        'seed': 2010,
    }
    '''##################################### OFFLINE #################################################'''
    pdim = 1
    nfolds = 5
    oof_train = np.zeros((ntrain, pdim))
    oof_test = np.zeros((len(data[data.day==24]), pdim))
    oof_test_folds = np.zeros((nfolds, len(data[data.day==24]), pdim))
    cv_scores = []

    kf = StratifiedKFold(nfolds, shuffle=True, random_state=2016)
    X, y = np.array(data.ix[((data.day>=18)&(data.day<=23)), features].values), np.array(data.ix[((data.day>=18)&(data.day<=23)), target].values).reshape(-1)
    X_test = np.array(data.ix[(data.day==24), features].values)
    for i, (train_ix, valid_ix) in enumerate(kf.split(X, y)):
        print('\n Fold {0}\n'.format(i + 1))
        X_train, X_val = X[train_ix], X[valid_ix]
        Y_train, Y_val = y[train_ix], y[valid_ix]

        d_train = lgb.Dataset(X_train, label=Y_train)
        d_valid = lgb.Dataset(X_val, label=Y_val)

        clf = lgb.train(params=params,
                        train_set=d_train,
                        valid_sets=d_valid,
                        num_boost_round=1200,
                        verbose_eval=20,
                        early_stopping_rounds=50)

        oof_train[valid_ix, :] = clf.predict(X_val).reshape(-1, pdim)
        oof_test_folds[i, :, :] = clf.predict(X_test).reshape(-1, pdim)

        src = log_loss(Y_val, oof_train[valid_ix, :])
        cv_scores.append(src)
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    print('cv_mean:  {}'.format(cv_mean))
    print('std_mean: {}'.format(cv_std))

    oof_test[:, :] = oof_test_folds.mean(axis=0)
    print('\nloss: ', log_loss(np.array(data.ix[(data.day==24), target].values).reshape(-1), oof_test.astype(float).reshape(-1,)))

    '''##################################### ONLINE #################################################'''
    # pdim = 1
    # nfolds = 5
    # oof_train = np.zeros((ntrain, pdim))
    # oof_test = np.zeros((ntest, pdim))
    # oof_test_folds = np.zeros((nfolds, ntest, pdim))
    # cv_scores = []
    #
    # kf = StratifiedKFold(nfolds, shuffle=True, random_state=2016)
    # X, y = np.array(data.ix[(data.day >= 17), features].values), np.array(data.ix[(data.day >= 17), target].values).reshape(-1)
    # X_test = np.array(test[features].values)
    # for i, (train_ix, valid_ix) in enumerate(kf.split(X, y)):
    #     print('\n Fold {0}\n'.format(i + 1))
    #     X_train, X_val = X[train_ix], X[valid_ix]
    #     Y_train, Y_val = y[train_ix], y[valid_ix]
    #
    #     d_train = lgb.Dataset(X_train, label=Y_train)
    #     d_valid = lgb.Dataset(X_val, label=Y_val)
    #
    #     clf = lgb.train(params=params,
    #                     train_set=d_train,
    #                     valid_sets=d_valid,
    #                     num_boost_round=1200,
    #                     verbose_eval=20,
    #                     early_stopping_rounds=50)
    #
    #     oof_train[valid_ix, :] = clf.predict(X_val).reshape(-1, pdim)
    #     oof_test_folds[i, :, :] = clf.predict(X_test).reshape(-1, pdim)
    #
    #     src = log_loss(Y_val, oof_train[valid_ix, :])
    #     cv_scores.append(src)
    # cv_mean = np.mean(cv_scores)
    # cv_std = np.std(cv_scores)
    #
    # print('cv_mean:  {}'.format(cv_mean))
    # print('std_mean: {}'.format(cv_std))
    #
    # oof_test[:, :] = oof_test_folds.mean(axis=0)
    #
    # csv_oof_train = pd.DataFrame(data=oof_train, columns=['predicted_score'])
    # csv_oof_test = pd.DataFrame(data=oof_test, columns=['predicted_score'])
    # csv_oof_test.insert(0, 'instance_id', test['instance_id'].values)
    # csv_oof_test.to_csv('baseline.txt', header=True, index=False, sep=' ')
