import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import random
import scipy.special as special

class BayesSmooth():
    def __init__(self, alpha, beta, df):
        self.alpha = alpha
        self.beta = beta
        self.df = df

    def sample_from_data(self, alpha, beta, num):
        I = []
        C = []
        for _ in range(num):
            imp = int(np.ceil(random.random() * self.df.shape[0]))
            I.append(imp)
            C.append(self.df.is_trade.sample(n=imp).sum())
        return I, C

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration'''
        for i in range(iter_num):
            if i%100 == 1:
                print('---{} iteration---'.format(i))
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        for i in range(len(tries)):
            sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
            sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
            sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

    def update_from_data_by_moment(self, tries, success):
        '''estimate alpha, beta using moment estimation'''
        mean, var = self.__compute_moment(tries, success)
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        #self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)

    def __compute_moment(self, tries, success):
        '''moment estimation'''
        ctr_list = []
        var = 0.0
        for i in range(len(tries)):
            ctr_list.append(float(success[i])/tries[i])
        mean = sum(ctr_list)/len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr-mean, 2)

        return mean, var/(len(ctr_list)-1)

    def export_ratio(self):
        s = self.sample_from_data(18,1000,10000)
        self.update_from_data_by_moment(s[0], s[1])
        s = self.sample_from_data(18,1000,10000)
        self.update_from_data_by_FPI(s[0], s[1], 1000, 0.00000001)
        return (self.df.is_trade.sum() + self.alpha) / (self.df.shape[0] + self.alpha + self.beta)

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
    check_time_difference = np.ones((len(df),1))
    num = {}
    timeseries = {}
    bd = df.day.min()
    for u, i, d in zip(df.user_id, df[behaviour], df.day):
        n += 1
        try:
            num[(u,i)] += 1
#            timeseries[(u,i)] = df.min_series_full[n-1] - timeseries[(u,i)]
            check_time_difference[n-1] = df.min_series_full[n-1] - timeseries[(u,i)]
            timeseries[(u,i)] = df.min_series_full[n-1]
        except:
            num[(u,i)] = 0
            timeseries[(u,i)] = df.min_series_full[n-1]
            check_time_difference[n-1] = 0

        check_time_day[n-1] = num[(u,i)]
        if d > bd:
            num = {}
        bd = d

    df['check_{}_min_difference'.format(behaviour)] = check_time_difference
    df['check_{}_time_day'.format(behaviour)] = check_time_day
    df['check_{}_ratio'.format(behaviour)] = df['check_{}_time_day'.format(behaviour)] / df['user_id_query_day_{}'.format(behaviour)]

    n = 0
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
        check_time_day_hour_map[n-1] = num[(u,i)]
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
    df['hour_series'] = (df.day-df.day.min()) * 4 + df.hour_map
    df['min_map'] = df['min'].apply(map_min)
    df['min_series'] = ((df.day-df.day.min()) * 24 + df.hour) * 4 + df.min_map
    df['min_series_full'] = ((df.day-df.day.min()) * 24 + df.hour) * 60 + df['min']
#    print(np.unique(df.hour_series))
#    print(np.unique(df.min_series))
    for f in ['user_id', 'item_id', 'shop_id', 'item_category_list', 'item_city_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'item_brand_id']:
        user_query_day = df.groupby([f, 'day']).size().reset_index().rename(columns={0: '{}_query_day'.format(f)})
        df = pd.merge(df, user_query_day, how = 'left', on=[f, 'day'])
        query_day_hour = df.groupby([f, 'day', 'hour']).size().reset_index().rename(columns={0: '{}_query_day_hour'.format(f)})
        df = pd.merge(df, query_day_hour, 'left',on=[f, 'day', 'hour'])
        query_day_hour_map = df.groupby([f, 'day', 'hour_map']).size().reset_index().rename(columns={0: '{}_query_day_hour_map'.format(f)})
        df = pd.merge(df, query_day_hour_map, 'left',on=[f, 'day', 'hour_map'])
        query_min = df.groupby([f, 'min_series']).size().reset_index().rename(columns={0: '{}_query_min_map'.format(f)})
        df = pd.merge(df, query_min, 'left',on=[f, 'min_series'])
    df['context_timestamp'] = df['context_timestamp'].apply(time2cov)
    df.sort_values(by='context_timestamp',inplace=True)

    for f in ['shop_id', 'item_brand_id', 'item_id', 'item_category_list','item_pv_level','item_sales_level','item_collected_level','item_price_level','context_page_id']: #,'predict_category_property_H_0','item_property_list_0']:
        df = user_check(df, f)

    n = 0
    check_time_day = np.ones((len(df),1))
    check_time_difference = np.ones((len(df),1))
    num = {}
    timeseries = {}
    bd = df.day.min()
    for u, d in zip(df.user_id, df.day):
        n += 1
        try:
            num[(u)] += 1
            timeseries[(u)] = df.min_series_full[n-1] - timeseries[(u)]
            check_time_difference[n-1] = timeseries[(u)]
        except:
            num[(u)] = 0
            timeseries[(u)] = df.min_series_full[n-1]
            check_time_difference[n-1] = 0

        check_time_day[n-1] = num[(u)]
        if d > bd:
            num = {}
        bd = d
    df['check_min_difference'] = check_time_difference
    df['check_time_day'] = check_time_day
    df['check_ratio_day_all'] = df['check_time_day'] / df['user_id_query_day']

    n = 0
    check_time_day_hour_map = np.ones((len(df),1))
    num = {}
    bd = df.day.min()
    bh = df.hour_map.min()
    for u, d, h in zip(df.user_id, df.day, df.hour_map):
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

    n = len(df)
    check_time_difference = np.ones((len(df),1))
    timeseries = {}
    for i in range(len(df)): #df.user_id[::-1]:
        u = df.user_id[n-i-1]
        try:
            check_time_difference[n-i-1] = timeseries[(u)]- df.min_series_full[n-i-1]
        except:
            check_time_difference[n-i-1] = 0
        timeseries[(u)] = df.min_series_full[n-i-1]

    df['check_min_difference_ahead'] = check_time_difference

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
