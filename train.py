from sklearn.preprocessing import StandardScaler
from utility import *
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from preprocessing import preKfold
from multiprocessing import Pool
import lightgbm as lgbm
import xgboost as xgb

def run(features, label, df, clf):
    print(features[-1])
    X = df
    #X['shop_review_positive_rate'] = np.log(X['shop_review_positive_rate'])
    #X['shop_review_positive_rate'] = np.log(X['shop_review_positive_rate'])
    #print(df.columns)
    y = df[label]
    kf = KFold(n_splits=3)
    Loss = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = preKfold(X.iloc[train_index], X.iloc[test_index], features)
        X_train, X_test = X_train[features], X_test[features]
        #norm = StandardScaler()
        #X_train = norm.fit_transform(X_train[features])
        #X_test = norm.transform(X_test[features])
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf.fit(X_train,y_train)
        predict = clf.predict_proba(X_test)[:,1]
        logloss = log_loss(y_test, predict)
        print(logloss)
        Loss.append(logloss)
    #for i,j in zip(features, clf.feature_importances_):
        #print('{}: {}'.format(i, j))
    print('Mean loss: {}'.format(np.mean(Loss)))
    #with open('log_lgb.log','a') as f:
        #f.write('\n{}\n{}\nMean loss: {}'.format('-'*30, features, np.mean(Loss)))
    return np.mean(Loss)

def run2(features, label, df, clf):
    print(features[-1])
    X = df
    y = df[label]
    Loss = []
    X_train, X_test = preKfold(X[X.context_timestamp <= '2018-09-23 23:59:59'], X[X.context_timestamp > '2018-09-23 23:59:59'], features)
    X_train, X_test = X_train[features], X_test[features]
    #norm = StandardScaler()
    #X_train = norm.fit_transform(X_train[features])
    #X_test = norm.transform(X_test[features])
    y_train, y_test = y[X.context_timestamp <= '2018-09-23 23:59:59'], y[X.context_timestamp > '2018-09-23 23:59:59']
    clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=True,early_stopping_rounds=100)
    predict = clf.predict_proba(X_test)[:,1]
    logloss = log_loss(y_test, predict)
    print(logloss)
    '''
    Loss.append(logloss)
    X_train, X_test = preKfold(X[X.context_timestamp >= '2018-09-18 23:59:59'], X[X.context_timestamp < '2018-09-18 23:59:59'], features)
    X_train, X_test = X_train[features], X_test[features]
    #norm = StandardScaler()
    #X_train = norm.fit_transform(X_train[features])
    #X_test = norm.transform(X_test[features])
    y_train, y_test = y[X.context_timestamp >= '2018-09-18 23:59:59'], y[X.context_timestamp < '2018-09-18 23:59:59']
    clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=True,early_stopping_rounds=100)
    predict = clf.predict_proba(X_test)[:,1]
    logloss = log_loss(y_test, predict)
    print(logloss)
    Loss.append(logloss)
    print('Mean loss: {}'.format(np.mean(Loss)))
    with open('log_xgb_f_select.log','a') as f:
        f.write('\n{}\n{}\nMean loss: {}'.format('-'*30, features, np.mean(Loss)))
    '''
    return np.mean(Loss)

def Myprediction(df, features, clf):
    testdf = pd.read_csv('data/test/round1_ijcai_18_test_a_20180301.txt',sep=' ')
    testdf.context_timestamp += 8*60*60
    testdf = convert_time(testdf)
    prediction_format = pd.read_csv('data/output/0203.txt',sep=' ')
    #testdf['sale_price'] = testdf['item_sales_level'] / testdf['item_price_level']
    train, predict = preKfold(df, testdf, features)
    clf.fit(train[features], train.is_trade, eval_set = [(train[features], train.is_trade)], eval_metric='logloss', verbose=True)
 #   print(predict[:3])
    prediction_format.predicted_score = clf.predict_proba(predict[features])[:,1]
 #   print(prediction_format[:3])
    prediction_format.to_csv('data/output/peter_0324.txt', sep=' ',index = None)

if __name__ == '__main__':
    #pool = Pool(8)
    df1 = pd.read_csv('data/train/train.csv')
    df1.context_timestamp += 8*60*60
    df1 = convert_time(df1)
    features = list(df1.columns)
    df = pd.read_csv('data/train/round1_ijcai_18_train_20180301.txt',sep=' ')
    df.context_timestamp += 8*60*60
    df = convert_time(df)
    print(df.shape)
    print(df.columns)
    uselessfeatures = ['instance_id', 'item_id', 'item_category_list', 'item_property_list','item_brand_id','item_city_id','user_id',
                     'context_id', 'context_timestamp', 'predict_category_property', 'shop_id', 'is_trade',
                      'item_property_list_0', 'item_property_list_1','item_property_list_2', 'item_property_list_3', 'item_property_list_4',
                      'predict_category_property_L_0', 'predict_category_property_H_0',
                      'predict_category_property_L_1', 'predict_category_property_H_1',
                      'predict_category_property_L_2', 'predict_category_property_H_2',
                      'predict_category_property_L_3', 'predict_category_property_H_3',
                      'predict_category_property_L_4', 'predict_category_property_H_4',
                      'item_category_list_0', 'item_category_list_1', 'item_category_list_2',
                      'item_category_list_3', 'item_category_list_4', 'item_category_list_5',
                      'item_category_list_6', 'item_category_list_7', 'item_category_list_8',
                      'item_category_list_9', 'item_category_list_10',
                      'item_category_list_11', 'item_category_list_12',
                      'item_category_list_13', 'user_occupation_id_0', 'user_occupation_id_1',
                      'user_occupation_id_2', 'user_occupation_id_3', 'user_occupation_id_4',
                      'user_gender_id_0', 'user_gender_id_1', 'user_gender_id_2',
                      'user_gender_id_3', 'user_age_level_0', 'user_age_level_1',
                      'user_age_level_2', 'user_age_level_3', 'user_age_level_4',
                      'user_age_level_5', 'user_age_level_6', 'user_age_level_7',
                      'user_age_level_8'
                      ]

    for uf in uselessfeatures:
        features.remove(uf)

    label = 'is_trade'
    add = []
    startscore = 1
    run_loop = True
    start_features = ['item_category_list-mean', 'shop_score_delivery', 'item_sales_level', 'hour', 'item_price_level', 'user_age_level', 'user_star_level', 
                      'item_collected_level', 'shop_star_level', 'item_pv_level', 'shop_review_positive_rate', 'context_page_id', 
                      'user_gender_id', 'user_age_level_7', 'item_category_list_7', 'user_occupation_id','user_id_query_day', 'user_id_query_day_hour',
                     ] #xgboost
#    start_features = ['item_category_list-mean', 'shop_score_delivery', 'item_sales_level', 'hour', 'item_price_level', 'user_age_level', 'user_star_level', 'item_collected_level', 'shop_star_level', 'item_pv_level', 'shop_review_positive_rate', 'context_page_id', 
#                      'user_gender_id', 'user_occupation_id','user_id_query_day', 'user_id_query_day_hour',]
#    start_features = ['item_price_level', 'item_sales_level', 'user_id_query_day', 'shop_score_service', 'hour', 'shop_id_query_day', 'shop_review_num_level', 'user_age_level', 'user_star_level', 'item_collected_level', 'item_city_id_query_day', 'user_id_query_day_hour', 'user_age_level_query_day', 'item_category_list-mean', 'context_page_id']

#    clf = xgb.XGBClassifier(seed = 1, max_depth=5, n_estimators=110) #Mean loss: 0.08743897882220214
    #for i in [0.1,0.05,0.01]:
    clf = lgbm.LGBMClassifier(random_state=1, num_leaves = 29, n_estimators=100,max_depth=-1,learning_rate = 0.1) #Mean loss: 0.08752755876530835
#    start_features = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
#                'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
#                'user_age_level', 'user_star_level', #'user_query_day', 'user_query_day_hour',
#                'context_page_id', 'hour', 'shop_id', 'shop_review_num_level', 'shop_star_level',
#                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
#                ]
    #run2(start_features, label, df, clf)
#    for nl in range(35,43,1):
#    for md in range(3,9):
#        nl = 63
#        print(nl)
#        print(md)
#        clf = lgbm.LGBMClassifier(random_state=1,num_leaves = nl, n_estimators=2000, max_depth=md, boosting_type='goss')
#        run2(start_features, label, df, clf)
    #print(features)
    '''
    while run_loop:
        run_loop = False
        start_features = start_features + add
        for f in features:
            if f not in start_features:
                print('-' * 30)
                #clf = xgb.XGBClassifier(seed=1, max_depth=3, n_estimators=100,nthread=-1)
                #clf = lgbm.LGBMClassifier(random_state=1,num_leaves = 63, n_estimators=2000) # used to be 29
                score = run2(start_features + [f], label, df, clf)
                if score <= startscore:
                    startscore = score
                    add = [f]
                    run_loop = True
    '''
    Myprediction(df, start_features, clf)
