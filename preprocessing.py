from utility import *
import time

replace_features = [['item_category_list'], ['item_city_id'],['item_price_level'],['item_sales_level'],['item_collected_level'],['item_pv_level'],
                    ['user_gender_id'], ['user_age_level'],['user_occupation_id'],['user_star_level'],
                    ['context_page_id'],
                    ['shop_review_num_level'], ['shop_star_level']
                    ]

one_hot_features = ['item_category_list','user_occupation_id','user_gender_id','user_age_level']

def preprocessing(traindf):
    start = time.time()
    traindf = convert_time(traindf)

#    for r_f in replace_features:
#        traindf = merge_mean(traindf,r_f)

    for ohf in one_hot_features:
        traindf = one_hot(traindf, ohf)

    get_first_n = 5
    name = 'item_property_list'
    add_list = pd.read_csv('{}.csv'.format(name))
    add_list['{}_ratio'.format(name)] = add_list.traded_quantity / add_list.total_quantity
    add_list.loc[add_list.total_quantity < 1000, '{}_ratio'.format(name)] = 0

    ipl = {}
    for j in range(len(add_list)):
        ipl[str(add_list.item_property_list_element[j])] = add_list.item_property_list_ratio[j]


    add = np.zeros((len(traindf),get_first_n))
    for index, i in enumerate(traindf[name]):
        jlist = []
        for j in i.split(';'):
            try:
                jlist.append([ipl[j],j])
            except:
                pass
        jlist = np.array(jlist)
        jlist = jlist[jlist[:,0].argsort()][::-1]
#        jlist = np.sort(jlist)[::-1]
        try:
            for i in range(get_first_n):
                add[index,i] = jlist[i,1]
        except:
            pass
    for i in range(get_first_n):
        traindf['{}_{}'.format(name,i)] = add[:,i]

    name = 'predict_category_property'
    add_list = pd.read_csv('{}_L.csv'.format(name))
    add_list1 = pd.read_csv('{}_H.csv'.format(name))
    add_list['{}_L_ratio'.format(name)] = add_list.traded_quantity / add_list.total_quantity
    add_list1['{}_H_ratio'.format(name)] = add_list1.traded_quantity / add_list1.total_quantity
    add_list.loc[add_list.total_quantity < 500, '{}_L_ratio'.format(name)] = 0
    add_list1.loc[add_list1.total_quantity < 500, '{}_ratio'.format(name)] = 0
    ipl = {}
    ipl1 = {}
    add = np.zeros((len(traindf),get_first_n))
    add1 = np.zeros((len(traindf),get_first_n))
    for j in range(len(add_list)):
        ipl[str(add_list.predict_category_property_element_L[j])] = add_list.predict_category_property_L_ratio[j]
    for j in range(len(add_list1)):
        ipl1[str(add_list1.predict_category_property_element_H[j])] = add_list1.predict_category_property_H_ratio[j]

    for index, i in enumerate(traindf[name]):
        jlist = []
        jlist1 = []
        for j in i.split(';'):
            t = j.split(':')
            try:
                jlist.append([ipl1[t[0]],ipl1[t[0]]])
            except:
                pass
            try:
                for k in t[1].split(','):
                    jlist1.append([ipl[k],k])
            except:
                pass
        jlist = np.array(jlist)
        jlist = jlist[jlist[:,0].argsort()][::-1]
        jlist1 = np.array(jlist1)
        try:
            jlist1 = jlist1[jlist1[:,0].argsort()][::-1]
        except:
            pass
#        jlist = np.sort(jlist)[::-1]
#        jlist1 = np.sort(jlist1)[::-1]
        try:
            for i in range(get_first_n):
                add[index,i] = jlist[i,1]
                add1[index,i] = jlist1[i,1]
        except:
            pass
    for i in range(get_first_n):
        traindf['{}_L_{}'.format(name, i)] = add1[:,i]
        traindf['{}_H_{}'.format(name, i)] = add[:,i]


    traindf.fillna(-1, inplace=True)
    print('Used time: {} min'.format(np.round((time.time() - start)/60),2))
    return traindf

def preKfold(train, test, features):
    start = time.time()
    for r_f in features:
        temp = r_f.split('-')
        if temp[-1] == 'mean':
            r_f = temp[:-1]
            train = merge_mean(train, r_f)
            featuresname = '-'.join(r_f) + '-mean'
            #print(train[r_f + [featuresname]].shape)
            #print(test.shape)
            #print(train.columns)
            #print(featuresname)
            #print(r_f + [featuresname])
            #print(test.columns)
            test = pd.merge(test, train[r_f + [featuresname]].drop_duplicates(), on = r_f, how = 'left')
            #print(test.shape)

    for ohf in one_hot_features:
        train = one_hot(train, ohf)
        test = one_hot(test, ohf)
    #train.fillna(-1, inplace=True)
    #test.fillna(-1, inplace=True)
    print('Used time: {} s'.format(np.round((time.time() - start)),2))
    return train, test

def item_property_list_element_obtain(df):
    C = {}
    CN = {}
    for index, i in enumerate(df['item_property_list']):
        for j in i.split(';'):
            C[j] =  C.get(j, 0) + df.is_trade[index]
            CN[j] =  CN.get(j, 0) + 1
    with open('item_property_list.csv','w') as f:
        f.write('item_property_list_element,traded_quantity,total_quantity\n')
        for key, val in C.items():
            f.write('{},{},{}\n'.format(key,val,CN[key]))

def predict_category_property_element_obtain(df):
    A = {}
    B = {}
    AN = {}
    BN = {}
    for index, i in enumerate(df['predict_category_property']):
        for j in i.split(';'):
            t = j.split(':')
            A[t[0]] =  A.get(t[0], 0) + df.is_trade[index]
            AN[t[0]] =  AN.get(t[0], 0) + 1
            try:
                for k in j.split(':')[1].split(','):
                    B[k] =  B.get(k, 0) + df.is_trade[index]
                    BN[k] =  BN.get(k, 0) + 1
            except:
                pass
    with open('predict_category_property_H.csv','w') as f:
        f.write('predict_category_property_element_H,traded_quantity,total_quantity\n')
        for key, val in A.items():
            f.write('{},{},{}\n'.format(key,val,AN[key]))

    with open('predict_category_property_L.csv','w') as f:
        f.write('predict_category_property_element_L,traded_quantity,total_quantity\n')
        for key, val in B.items():
            f.write('{},{},{}\n'.format(key,val,BN[key]))


if __name__ == "__main__":
    #traindf = pd.read_csv('data/train/round1_ijcai_18_train_20180301.txt',sep=' ')
    #item_property_list_element_obtain(traindf)
    #predict_category_property_element_obtain(traindf)
    #df = preprocessing(traindf)
    #df.to_csv('data/train/train.csv', index = None)
    testdf = pd.read_csv('data/test/round1_ijcai_18_test_a_20180301.txt',sep=' ')
    df = preprocessing(testdf)
    df.to_csv('data/test/test.csv', index = None)
