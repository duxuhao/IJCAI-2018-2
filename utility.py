import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

def visualize_feature(featurename, df, visual = 1):
    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot(111)
    num = len(np.unique(df[featurename]))
    if np.sum(df[featurename] == -1):
        num -= 1
    if num > 100:
        num=100
    n1, bins, patches = plt.hist(df[(df.is_trade == 0) & (df[featurename] != -1)][featurename],
                                 num,label = 'not traded')
    n2, bins, patches = plt.hist(df[(df.is_trade == 1) & (df[featurename] != -1)][featurename],
                                 bins = bins,label = 'traded')
    plt.legend(fontsize=14)
    plt.xlabel(featurename,fontsize=14)
    plt.ylabel('Quantity',fontsize=14)
    ax.set_yscale('log')
    ax.tick_params('y', colors='r')
    ax2 = ax.twinx()
    ratio = n2/n1
    ratio = np.nan_to_num(ratio)
    ax2.plot((bins[:-1]+bins[1:])/2, ratio, 'k*-',linewidth=3)
    ax2.set_ylabel('Ratio', color='k')
    ax2.tick_params('y', colors='k')
    if visual == 1:
        plt.show()
    else:
        plt.close()
    try:
        print(df[featurename].std()/np.sqrt(np.mean(df[featurename] ** 2)))
    except:
        pass
    return n2/n1

def evaluate(true,pred):
    return -np.mean(true * np.log10(pred) + (1-true) * np.log10(1-pred))

def merge_mean(df, features):
    t = df.groupby(features)['is_trade'].mean().reset_index()
    t.rename(columns={'is_trade': '-'.join(features) + '-mean'}, inplace=True)
    return pd.merge(df, t, on = features, how = 'left')

def merge_std(df, features):
    t = df.groupby(features)['is_trade'].mean().reset_index()
    t.rename(columns={'is_trade': '-'.join(features) + '-std'}, inplace=True)
    return pd.merge(df, t, on = features, how = 'left')

def merge_median(df, features):
    t = df.groupby(features)['is_trade'].mean().reset_index()
    t.rename(columns={'is_trade': '-'.join(features) + '-median'}, inplace=True)
    return pd.merge(df, t, on = features, how = 'left')

def merge_multiple(df, feature1, feature2):
    df['{}*{}'.format(feature1,feature2)] = df[feature1] * df[feature2]
    return df

def merge_divide(df, feature1, feature2):
    df['{}/{}'.format(feature1,feature2)] = (df[feature1]+0.001) / (df[feature2]+0.001)
    return df

def one_hot(df, feature):
    t = pd.get_dummies(df[feature])
    t.columns = ['{}_{}'.format(feature,i) for i in range(len(t.columns))]
    return pd.concat([df, t], axis=1)

def map_hour(x):
    if (x >= 7) & (x <= 12):
        return 1
    elif (x >= 13) & (x <= 17):
        return 2
    elif (x > 17) & (x <= 24):
        return 3
    else:
        return 4

def map_min(x):
    return int(x/15)

def time2cov(time_):
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_))

def user_check(df, behaviour):
    user_day = df.groupby(['user_id', 'day', behaviour]).size().reset_index().rename(columns={0: 'user_id_query_day_{}'.format(behaviour)})
    df = pd.merge(df, user_day, how = 'left', on=['user_id', 'day',behaviour])
    user_day_hour = df.groupby(['user_id', 'day', 'hour', behaviour]).size().reset_index().rename(columns={0: 'user_id_query_day_hour_{}'.format(behaviour)})
    df = pd.merge(df, user_day_hour, how = 'left', on=['user_id', 'day', 'hour',behaviour])
    user_day_hour_map = df.groupby(['user_id', 'day', 'hour_map', behaviour]).size().reset_index().rename(columns={0: 'user_id_query_day_hour_map_{}'.format(behaviour)})
    df = pd.merge(df, user_day_hour_map, how = 'left', on=['user_id', 'day', 'hour_map',behaviour])
    n = 0
    check_time_day = np.ones((len(df),1))
    num = {}
    bd = df.day.min()
    for u, i, d in zip(df.user_id, df[behaviour], df.day):
        n += 1
        try:
            num[(u,i)] += 1
        except:
            num[(u,i)] = 0
        check_time_day[n-1] = num[(u,i)]
        if d > bd:
            num = {}
        bd = d
    df['check_{}_time_day'.format(behaviour)] = check_time_day
    df['check_{}_ratio'.format(behaviour)] = df['check_{}_time_day'.format(behaviour)] / df['user_id_query_day_{}'.format(behaviour)]

    check_time_day_hour_map = np.ones((len(df),1))
    num = {}
    bd = df.day.min()
    bh = df.hour_map.min()
    for u, i, d, h in zip(df.user_id, df[behaviour], df.day, df.hour_map):
        n += 1
        try:
            num[(u,i)] += 1
        except:
            num[(u,i)] = 0
        check_time_day_map[n-1] = num[(u,i)]
        if (d > bd) | (h > bh):
            num = {}
        bd = d
        bh = h
    df['check_{}_time_day_hour_map'.format(behaviour)] = check_time_day_hour_map
    df['check_{}_ratio_hour_map'.format(behaviour)] = df['check_{}_time_day_hour_map'.format(behaviour)] / df['user_id_query_day_hour_map_{}'.format(behaviour)]
    return df

def convert_time(df):
    df['hour'] = [int(datetime.datetime.fromtimestamp(i).strftime('%H')) for i in df.context_timestamp]
    df['day'] = [int(datetime.datetime.fromtimestamp(i).strftime('%d')) for i in df.context_timestamp]
    df['min'] = [int(datetime.datetime.fromtimestamp(i).strftime('%M')) for i in df.context_timestamp]
    df['hour_map'] = df['hour'].apply(map_hour)
    df['hour_series'] = (df.day-df.day.min()) * 24 + df.hour)
    df['min_map'] = df['min'].apply(map_min)
    df['min_series'] = ((df.day-df.day.min()) * 24 + df.hour) * 4 + df.min_map
    for f in ['user_id', 'item_id', 'shop_id', 'item_category_list', 'item_city_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'item_brand_id']:
        user_query_day = df.groupby([f, 'day']).size().reset_index().rename(columns={0: '{}_query_day'.format(f)})
        df = pd.merge(df, user_query_day, how = 'left', on=[f, 'day'])
        query_day_hour = df.groupby([f, 'day', 'hour']).size().reset_index().rename(columns={0: '{}_query_day_hour'.format(f)})
        df = pd.merge(df, query_day_hour, 'left',on=[f, 'day', 'hour'])
        query_day_hour_map = df.groupby([f, 'day', 'hour_map']).size().reset_index().rename(columns={0: '{}_query_day_hour_map'.format(f)})
        df = pd.merge(df, query_day_hour_map, 'left',on=[f, 'day', 'hour_map'])
        query_min = df.groupby([f, 'min_series']).size().reset_index().rename(columns={0: '{}_query_min_map'.format(f)})
        df = pd.merge(df, query_min, 'left',on=[f, 'min_map'])
    df['context_timestamp'] = df['context_timestamp'].apply(time2cov)
    df.sort_values(by='context_timestamp',inplace=True)

    for f in ['shop_id', 'item_brand_id', 'item_id', 'item_category_list','item_pv_level','item_sales_level','item_collected_level','item_price_level','context_page_id','predict_category_property_H_0','item_property_list_0']:
        try:
            df = user_check(df, f)
        except:
            pass
    n = 0
    check_time_day = np.ones((len(df),1))
    num = {}
    bd = df.day.min()
    for u, d in zip(df.user_id, df.day):
        n += 1
        try:
            num[(u)] += 1
        except:
            num[(u)] = 0
        check_time_day[n-1] = num[(u)]
        if d > bd:
            num = {}
        bd = d
    df['check_time_day'] = check_time_day
    df['check_ratio_day_all'] = df['check_time_day'] / df['user_id_query_day']

    n = 0
    check_time_day_hour_map = np.ones((len(df),1))
    num = {}
    bd = df.day.min()
    bh = df.hour_map.min()
    for u, d in zip(df.user_id, df.day, df.hour_map):
        n += 1
        try:
            num[(u)] += 1
        except:
            num[(u)] = 0
        check_time_day_hour_map[n-1] = num[(u)]
        if (d > bd) | (h > bh):
            num = {}
        bd = d
        bh = h
    df['check_time_day_hour_map'] = check_time_day
    df['check_ratio_day_hour_map_all'] = df['check_time_day_hour_map'] / df['user_id_query_day_hour_map']
    return df

def preKfold(train, test, features):
    for r_f in features:
        temp = r_f.split('-')
        if temp[-1] == 'mean':
            train = merge_mean(train, temp[:-1])
            test = pd.merge(test, train[temp[:-1] + [r_f]].drop_duplicates(), on = temp[:-1], how = 'left')
        elif temp[-1] == 'std':
            train = merge_mean(train, temp[:-1])
            test = pd.merge(test, train[temp[:-1] + [r_f]].drop_duplicates(), on = temp[:-1], how = 'left')

    return train, test
