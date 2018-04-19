import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn import preprocessing
import warnings
from utility import convert_time
from multiprocessing import Pool

warnings.filterwarnings("ignore")

import time


def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


def base_process(data):
    lbl = preprocessing.LabelEncoder()
    print(
        '--------------------------------------------------------------item--------------------------------------------------------------')
    data['len_item_category'] = data['item_category_list'].map(lambda x: len(str(x).split(';')))
    data['len_item_property'] = data['item_property_list'].map(lambda x: len(str(x).split(';')))
    for i in range(1, 3):
        data['item_category_list' + str(i)] = lbl.fit_transform(data['item_category_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))  # item_category_list的第0列全部都一样
    for i in range(10):
        data['item_property_list' + str(i)] = lbl.fit_transform(data['item_property_list'].map(lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    for col in ['item_id', 'item_brand_id', 'item_city_id']:
        data[col] = lbl.fit_transform(data[col])
    print(
        '--------------------------------------------------------------user--------------------------------------------------------------')
    for col in ['user_id']:
        data[col] = lbl.fit_transform(data[col])
    print('user 0,1 feature')
    data['gender0'] = data['user_gender_id'].apply(lambda x: 1 if x == -1 else 2)
    data['age0'] = data['user_age_level'].apply(lambda x: 1 if x == 1004 | x == 1005 | x == 1006 | x == 1007  else 2)
    data['occupation0'] = data['user_occupation_id'].apply(lambda x: 1 if x == -1 | x == 2003  else 2)
    data['star0'] = data['user_star_level'].apply(lambda x: 1 if x == -1 | x == 3000 | x == 3001  else 2)
    print(
        '--------------------------------------------------------------context--------------------------------------------------------------')
    data = convert_time(data)
    data['len_predict_category_property'] = data['predict_category_property'].map(lambda x: len(str(x).split(';')))
    for i in range(5):
        data['predict_category_property' + str(i)] = lbl.fit_transform(data['predict_category_property'].map(
            lambda x: str(str(x).split(';')[i].split(':')[0]) if len(str(x).split(';')) > i else ''))
    print('context 0,1 feature')
    data['context_page0'] = data['context_page_id'].apply(
        lambda x: 1 if x == 4001 | x == 4002 | x == 4003 | x == 4004 | x == 4007  else 2)
    print(
        '--------------------------------------------------------------shop--------------------------------------------------------------')
    for col in ['shop_id']:
        data[col] = lbl.fit_transform(data[col])
    data['shop_score_delivery0'] = data['shop_score_delivery'].apply(lambda x: 0 if x <= 0.98 and x >= 0.96  else 1)
    return data

def deliver(x):
    #x=round(x,6)
    jiange=0.1
    for i in range(1,20):
        if (x>=4.1+jiange*(i-1))&(x<=4.1+jiange*i):
            return i+1
    if x==-5:
        return 1

def deliver1(x):
    if (x>=2)&(x<=4):
        return 1
    elif (x>=5)&(x<=7):
        return 2
    else:
        return 3


def review(x):
    # x=round(x,6)
    jiange = 0.02
    for i in range(1, 30):
        if (x >= 0.714 + jiange * (i - 1)) & (x <= 0.714 + jiange * i):
            return i + 1
    if x == -1:
        return 1

def review1(x):
    # x=round(x,6)
    if (x>=2)&(x<=12):
        return 1
    elif (x>=13)&(x<=15):
        return 2
    else:
        return 3


def service(x):
    #x=round(x,6)
    jiange=0.1
    for i in range(1,20):
        if (x>=3.93+jiange*(i-1))&(x<=3.93+jiange*i):
            return i+1
    if x==-1:
        return 1

def service1(x):
    if (x>=2)&(x<=7):
        return 1
    elif (x>=8)&(x<=9):
        return 2
    else:
        return 3


def describe(x):
    #x=round(x,6)
    jiange=0.1
    for i in range(1,30):
        if (x>=3.93+jiange*(i-1))&(x<=3.93+jiange*i):
            return i+1
    if x==-1:
        return 1

def describe1(x):
    if (x>=2)&(x<=8):
        return 1
    elif (x>=9)&(x<=10):
        return 2
    else:
        return 3

def shop_fenduan(data):
    data['shop_score_delivery'] = data['shop_score_delivery'] * 5
    data = data[data['shop_score_delivery'] != -5]
    data['deliver_map'] = data['shop_score_delivery'].apply(deliver)
    data['deliver_map'] = data['deliver_map'].apply(deliver1)
    # del data['shop_score_delivery']
    print(data.deliver_map.value_counts())

    data['shop_score_service'] = data['shop_score_service'] * 5
    data = data[data['shop_score_service'] != -5]
    data['service_map'] = data['shop_score_service'].apply(service)
    data['service_map'] = data['service_map'].apply(service1)
    # del data['shop_score_service']
    print(data.service_map.value_counts())  # 视为好评，中评，差评
    #
    data['shop_score_description'] = data['shop_score_description'] * 5
    data = data[data['shop_score_description'] != -5]
    data['de_map'] = data['shop_score_description'].apply(describe)
    data['de_map'] = data['de_map'].apply(describe1)
    # del data['shop_score_description']
    print(data.de_map.value_counts())

    data = data[data['shop_review_positive_rate'] != -1]
    data['review_map'] = data['shop_review_positive_rate'].apply(review)
    data['review_map'] = data['review_map'].apply(review1)
    print(data.review_map.value_counts())

    data['normal_shop'] = data.apply(
        lambda x: 1 if (x.deliver_map == 3) & (x.service_map == 3) & (x.de_map == 3) & (x.review_map == 3) else 0,
        axis=1)
    del data['de_map']
    del data['service_map']
    del data['deliver_map']
    del data['review_map']
    return data

def get_trade(df1,df2,post):
    fea_list = ['instance_id']
    for m in ['user_id','user_age_level','user_gender_id','user_occupation_id']:
        print(m)
        trd = df1.groupby(by=m).sum()['is_trade'].reset_index()
        trd2 = df1.groupby(by=m).mean()['is_trade'].reset_index()
        fn = '{}_trade{}'.format(m, post)
        fn2 = '{}_trade_mean{}'.format(m, post)
        trd.rename(index=str, columns={"is_trade": fn}, inplace = True)
        trd2.rename(index=str, columns={"is_trade": fn2}, inplace = True)
        df2 = df2.merge(trd, on = [m], how = 'left')
        df2 = df2.merge(trd2, on = [m], how = 'left')
        fea_list.append(fn)
        fea_list.append(fn2)
        for n in ['item_id','item_category_list', 'shop_id', 'item_brand_id', 'item_price_level','item_sales_level', 'item_pv_level', 'shop_review_num_level', 'item_city_id']:
            trd = df1.groupby(by=[m] + [n]).sum()['is_trade'].reset_index() #.to_dict()
            trd2 = df1.groupby(by=[m] + [n]).mean()['is_trade'].reset_index()
            fn = '{}_{}_trade{}'.format(m, n, post)
            fn2 = '{}_{}_trade_mean{}'.format(m, n, post)
            trd.rename(index=str, columns={"is_trade": fn}, inplace = True)
            trd2.rename(index=str, columns={"is_trade": fn2}, inplace = True)
            df2 = df2.merge(trd, on = [m] + [n], how = 'left')
            df2 = df2.merge(trd2, on = [m] + [n], how = 'left')
            fea_list.append(fn)
            fea_list.append(fn2)
    df2 = df2[fea_list]
    return df2

def slice_trade(data):
    for index, t in enumerate(range(19,26)):
        print(t)
        # 19到25，25是test
        df1 = data[data['day'] < t]
        df2 = data[data['day'] == t]
        df2 = get_trade(df1, df2,'xd')
        if index == 0:
            Df2 = df2
        else:
            Df2 = pd.concat([df2, Df2])
    data = pd.merge(data, Df2, on=['instance_id'], how='left')

    return data

def get_cnt(df1, df2, post):
    fea_list = ['instance_id']
    for f in ['user_id', 'item_id', 'shop_id', 'item_city_id', 'item_brand_id', 'item_property_list0', 'item_category_list']:
        cnt = df1.groupby(by=f).count()['instance_id'].to_dict()
        fn = '{}_cnt{}'.format(f, post)
        df2[fn] = df2[f].apply(lambda x: cnt.get(x, 0))
        fea_list.append(fn)
    df2 = df2[fea_list]
    return df2

def slide_cnt(data):
    for TimeSeries in ['day', 'hour_series']:
#        print(TimeSeries)
        for index, t in enumerate(range(data[TimeSeries].min() + 1, data[TimeSeries].max() + 1)):  # 18到24号
            df1 = data[data[TimeSeries] == t - 1]
            df2 = data[data[TimeSeries] == t]  # 19到25号
            df2 = get_cnt(df1, df2,'1{}'.format(TimeSeries[0]))
            if index == 0:
                Df2 = df2
            else:
                Df2 = pd.concat([df2, Df2])
        data = pd.merge(data, Df2, on=['instance_id'], how='left')
        for index, t in enumerate(range(data[TimeSeries].min() + 1, data[TimeSeries].max() + 1)):
#            print(t)
            # 19到25，25是test
            df1 = data[data['day'] < t]
            df2 = data[data['day'] == t]
            df2 = get_cnt(df1, df2,'x{}'.format(TimeSeries[0]))
            if index == 0:
                Df2 = df2
            else:
                Df2 = pd.concat([df2, Df2])
        data = pd.merge(data, Df2, on=['instance_id'], how='left')

    return data


def zuhe(data):
    for col in ['user_gender_id','user_age_level','user_occupation_id','user_star_level']:
        data[col] = data[col].apply(lambda x: 0 if x == -1 else x)

    for col in ['item_sales_level', 'item_price_level', 'item_collected_level',
                'user_gender_id','user_age_level','user_occupation_id','user_star_level',
                'shop_review_num_level', 'shop_star_level']:
        data[col] = data[col].astype(str)

    print('item两两组合')
    data['sale_price'] = data['item_sales_level'] + data['item_price_level']
    data['sale_collect'] = data['item_sales_level'] + data['item_collected_level']
    data['price_collect'] = data['item_price_level'] + data['item_collected_level']

    print('user两两组合')
    data['gender_age'] = data['user_gender_id'] + data['user_age_level']
    data['gender_occ'] = data['user_gender_id'] + data['user_occupation_id']
    data['gender_star'] = data['user_gender_id'] + data['user_star_level']

    print('shop两两组合')
    data['review_star'] = data['shop_review_num_level'] + data['shop_star_level']


    for col in ['item_sales_level', 'item_price_level', 'item_collected_level',  'sale_price','sale_collect', 'price_collect',
                'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level','gender_age','gender_occ','gender_star',
                'shop_review_num_level','shop_star_level','review_star']:
        data[col] = data[col].astype(int)

    del data['review_star']

    return data

def item(data):
    print('一个item有多少brand,price salse collected level……')

    itemcnt = data.groupby(['item_id'], as_index=False)['instance_id'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_id'], how='left')

    for col in ['user_id','item_brand_id','item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_id'], as_index=False)['instance_id'].agg({str(col) + '_item_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_id'], how='left')
        data[str(col) + '_item_prob']=data[str(col) + '_item_cnt']/data['item_cnt']
    del data['item_cnt']

    print('一个brand有多少price salse collected level……')

    itemcnt = data.groupby(['item_brand_id'], as_index=False)['instance_id'].agg({'item_brand_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_brand_id'], how='left')

    for col in ['user_id','item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_brand_id'], as_index=False)['instance_id'].agg({str(col) + '_brand_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_brand_id'], how='left')
        data[str(col) + '_brand_prob'] = data[str(col) + '_brand_cnt'] / data['item_brand_cnt']
    del data['item_brand_cnt']

    print('一个city有多少item_price_level，item_sales_level，item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_city_id'], as_index=False)['instance_id'].agg({'item_city_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_city_id'], how='left')
    for col in ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_city_id'], as_index=False)['instance_id'].agg({str(col) + '_city_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_city_id'], how='left')
        data[str(col) + '_city_prob'] = data[str(col) + '_city_cnt'] / data['item_city_cnt']
    del data['item_city_cnt']

    print('一个price有多少item_sales_level，item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_price_level'], as_index=False)['instance_id'].agg({'item_price_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_price_level'], how='left')
    for col in ['user_id','item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_city_id'], as_index=False)['instance_id'].agg({str(col) + '_price_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_city_id'], how='left')
        data[str(col) + '_price_prob'] = data[str(col) + '_price_cnt'] / data['item_price_cnt']
    del data['item_price_cnt']

    print('一个item_sales_level有多少item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_sales_level'], as_index=False)['instance_id'].agg({'item_salse_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_sales_level'], how='left')
    for col in ['user_id','item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_sales_level'], as_index=False)['instance_id'].agg({str(col) + '_salse_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_sales_level'], how='left')
        data[str(col) + '_salse_prob'] = data[str(col) + '_salse_cnt'] / data['item_salse_cnt']
    del data['item_salse_cnt']

    print('一个item_collected_level有多少item_pv_level')

    itemcnt = data.groupby(['item_collected_level'], as_index=False)['instance_id'].agg({'item_coll_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_collected_level'], how='left')
    for col in ['user_id','item_pv_level']:
        itemcnt = data.groupby([col, 'item_collected_level'], as_index=False)['instance_id'].agg({str(col) + '_coll_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_collected_level'], how='left')
        data[str(col) + '_coll_prob'] = data[str(col) + '_coll_cnt'] / data['item_coll_cnt']
    del data['item_coll_cnt']

    return data

def user(data):
    print('用户有多少性别')
    itemcnt = data.groupby(['user_id'], as_index=False)['instance_id'].agg({'user_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_id'], how='left')

    for col in ['user_gender_id','user_age_level', 'user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg({str(col) + '_user_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob']=data[str(col) + '_user_cnt']/data['user_cnt']
    del data['user_cnt']

    print('性别的年龄段，职业有多少')
    itemcnt = data.groupby(['user_gender_id'], as_index=False)['instance_id'].agg({'user_gender_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_gender_id'], how='left')

    for col in ['user_age_level', 'user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg({str(col) + '_user_gender_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob']=data[str(col) + '_user_gender_cnt']/data['user_gender_cnt']
    del data['user_gender_cnt']

    print('user_age_level对应的user_occupation_id，user_star_level')
    itemcnt = data.groupby(['user_age_level'], as_index=False)['instance_id'].agg({'user_age_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_age_level'], how='left')

    for col in ['user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg({str(col) + '_user_age_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob']=data[str(col) + '_user_age_cnt']/data['user_age_cnt']
    del data['user_age_cnt']

    print('user_occupation_id对应的user_star_level')
    itemcnt = data.groupby(['user_occupation_id'], as_index=False)['instance_id'].agg({'user_occ_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_occupation_id'], how='left')
    for col in ['user_star_level']:
        itemcnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg({str(col) + '_user_occ_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob']=data[str(col) + '_user_occ_cnt']/data['user_occ_cnt']
    del data['user_occ_cnt']

    return data

def user_item(data):
    itemcnt = data.groupby(['user_id'], as_index=False)['instance_id'].agg({'user_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_id'], how='left')
    print('一个user有多少item_id,item_brand_id……')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg({str(col)+'_user_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob'] = data[str(col) + '_user_cnt'] / data['user_cnt']

    print('一个user_gender有多少item_id,item_brand_id……')
    itemcnt = data.groupby(['user_gender_id'], as_index=False)['instance_id'].agg({'user_gender_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_gender_id'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg({str(col)+'_user_gender_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob'] = data[str(col) + '_user_gender_cnt'] / data['user_gender_cnt']

    print('一个user_age_level有多少item_id,item_brand_id……')
    itemcnt = data.groupby(['user_age_level'], as_index=False)['instance_id'].agg({'user_age_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_age_level'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg({str(col)+'_user_age_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob'] = data[str(col) + '_user_age_cnt'] / data['user_age_cnt']

    print('一个user_occupation_id有多少item_id,item_brand_id…')
    itemcnt = data.groupby(['user_occupation_id'], as_index=False)['instance_id'].agg({'user_occ_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_occupation_id'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg({str(col)+'_user_occ_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob'] = data[str(col) + '_user_occ_cnt'] / data['user_occ_cnt']

    return data


def user_shop(data):
    print('一个user有多少shop_id,shop_review_num_level……')

    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob'] = data[str(col) + '_user_cnt'] / data['user_cnt']
    del data['user_cnt']

    print('一个user_gender有多少shop_id,shop_review_num_level……')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_gender_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob'] = data[str(col) + '_user_gender_cnt'] / data['user_gender_cnt']
    del data['user_gender_cnt']

    print('一个user_age_level有多少shop_id,shop_review_num_level……')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_age_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob'] = data[str(col) + '_user_age_cnt'] / data['user_age_cnt']
    del data['user_age_cnt']

    print('一个user_occupation_id有多少shop_id,shop_review_num_level……')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_occ_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob'] = data[str(col) + '_user_occ_cnt'] / data['user_occ_cnt']
    del data['user_occ_cnt']

    return data


def shop_item(data):
    print('一个shop有多少item_id,item_brand_id,item_city_id,item_price_level……')
    itemcnt = data.groupby(['shop_id'], as_index=False)['instance_id'].agg({'shop_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['shop_id'], how='left')
    for col in ['user_id', 'item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'shop_id'], as_index=False)['instance_id'].agg({str(col)+'_shop_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'shop_id'], how='left')
        data[str(col) + '_shop_prob'] = data[str(col) + '_shop_cnt'] / data['shop_cnt']
    del data['shop_cnt']

    print('一个shop_review_num_level有多少item_id,item_brand_id,item_city_id,item_price_level……')
    itemcnt = data.groupby(['shop_review_num_level'], as_index=False)['instance_id'].agg({'shop_rev_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['shop_review_num_level'], how='left')
    for col in ['user_id', 'item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'shop_review_num_level'], as_index=False)['instance_id'].agg({str(col)+'_shop_rev_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'shop_review_num_level'], how='left')
        data[str(col) + '_shop_rev_prob'] = data[str(col) + '_shop_rev_cnt'] / data['shop_rev_cnt']
    del data['shop_rev_cnt']

    # print('一个shop_star_level有多少item_id,item_brand_id,item_city_id,item_price_level……')
    # itemcnt = data.groupby(['shop_star_level'], as_index=False)['instance_id'].agg({'shop_star_cnt': 'count'})
    # data = pd.merge(data, itemcnt, on=['shop_star_level'], how='left')
    # for col in ['item_id',
    #             'item_brand_id', 'item_city_id', 'item_price_level',
    #             'item_sales_level', 'item_collected_level', 'item_pv_level']:
    #     item_shop_cnt = data.groupby([col, 'shop_star_level'], as_index=False)['instance_id'].agg({str(col) + '_shop_star_cnt': 'count'})
    #     data = pd.merge(data, item_shop_cnt, on=[col, 'shop_star_level'], how='left')
    #     data[str(col) + '_shop_star_prob'] = data[str(col) + '_shop_star_cnt'] / data['shop_star_cnt']
    # del data['shop_star_cnt']
    return data

if __name__ == "__main__":
    train = pd.read_csv('data/train/round1_ijcai_18_train_20180301.txt',sep=' ')
    test = pd.read_csv('data/test/round1_ijcai_18_test_a_20180301.txt',sep=' ')
    test2 = pd.read_csv('data/test/round1_ijcai_18_test_b_20180418.txt',sep = ' ')
    train.context_timestamp += 8 * 60 * 60
    test.context_timestamp += 8 * 60 * 60
    test2.context_timestamp += 8 * 60 * 60
    train['used'] = 'a'
    test['used'] = 'a'
    test2['used'] = 'b'
    pool = Pool(8)
    data = pd.concat([train, test, test2])
    data = data.drop_duplicates(subset='instance_id')  # 把instance id去重
#    data['day'] = [int(datetime.datetime.fromtimestamp(i).strftime('%d')) for i in data.context_timestamp]
    print('make feature')
    data = base_process(data)
    data=shop_fenduan(data)
    data = slide_cnt(data)
    data = slice_trade(data)
    print('finish trade count')
    data = zuhe(data)
    print('----------------------------全局统计特征---------------------------------------------------')
    data = item(data)
    data = user(data)
    data = user_item(data)
    data = user_shop(data)
    data=shop_item(data)
    "----------------------------------------------------线下----------------------------------------"
#    item_category_list_unique = list(np.unique(data.item_category_list))
#    data.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)
    data.to_csv('data/train/trainb.csv',index = None)
