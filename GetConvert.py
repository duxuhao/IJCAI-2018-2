import pandas as pd
import numpy as np
import scipy.special as special
import random

df2 = pd.read_csv('data/train/train2.csv')
df2 = df2[~pd.isnull(df2.is_trade)]
item_category_list_unique = list(np.unique(df2.item_category_list))
df2.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)

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
            if i%100 == 0:
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
        self.update_from_data_by_FPI(s[0], s[1], 300, 0.00000001)
        return (self.df.is_trade.sum() + self.alpha) / (self.df.shape[0] + self.alpha + self.beta)

def convert(grouplist = ['user_gender_id','user_age_level'], df2):
    newconvert = []
    if 1:
        for g in np.unique(df2[grouplist[0]]):
            for a in np.unique(df2[grouplist[1]]):
                testdf = df2[(df2[grouplist[0]] == g) & (df2[grouplist[1]] == a)][grouplist + ['is_trade']]
                print(testdf.shape)
                t = BayesSmooth(1,1,testdf)
                if testdf.shape[0]:
                    newconvert.append([g,a,t.export_ratio()])
                else:
                    newconvert.append([g,a,0])

    convert = df2[grouplist + ['is_trade']].groupby(grouplist).mean().reset_index()#.rename({'is_trade': '_'.join(grouplist)}, inplace = True)
    newconvert = pd.DataFrame(np.array(newconvert))
    newconvert.columns = convert.columns
    T = pd.merge(newconvert, convert, on = grouplist)
    T.columns = [grouplist[0],grouplist[1],'{}_{}_mean_bayes'.format(grouplist[0],grouplist[1]),'{}_{}_mean_no_bayes'.format(grouplist[0],grouplist[1])]
    T.to_csv('{}_{}_mean.csv'.format(grouplist[0],grouplist[1]), index = None)

if __name__ == "__main__":
    df2 = pd.read_csv('data/train/train2.csv')
    df2 = df2[~pd.isnull(df2.is_trade)]
    item_category_list_unique = list(np.unique(df2.item_category_list))
    df2.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)
    convert(grouplist = ['user_gender_id','user_age_level'],df2 = df2)
    convert(grouplist = ['user_age_level','hour'], df2 = df2)
