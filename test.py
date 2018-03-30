import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

testdf = pd.read_csv('round1_ijcai_18_test_a_20180301.txt',sep=' ')
testdf['shop_review_positive_rate_level'] = 3
testdf.loc[testdf.shop_review_positive_rate < 0.995,'shop_review_positive_rate_level'] = 2
testdf.loc[testdf.shop_review_positive_rate < 0.99,'shop_review_positive_rate_level'] = 1
testdf.loc[testdf.shop_review_positive_rate < 0.97,'shop_review_positive_rate_level'] = 0
#print(df.columns)
#print(df.shape)
#a = list(np.unique(df.item_id))
traindf = pd.read_csv('round1_ijcai_18_train_20180301.txt',sep=' ')
traindf['shop_review_positive_rate_level'] = 3
traindf.loc[traindf.shop_review_positive_rate < 0.995,'shop_review_positive_rate_level'] = 2
traindf.loc[traindf.shop_review_positive_rate < 0.99,'shop_review_positive_rate_level'] = 1
traindf.loc[traindf.shop_review_positive_rate < 0.97,'shop_review_positive_rate_level'] = 0
#print(df.columns)
#print(df.shape)
#print(len(np.unique(df.item_id)))
#print(len(np.unique(a + list(np.unique(df.item_id)))))
for i in range(1,21):
    print(np.percentile(traindf.shop_review_positive_rate,i*5))
plt.hist(np.log(1.1-traindf.shop_review_positive_rate), 50, normed=1, facecolor='g', alpha=0.75)
plt.show()
col = ['item_id','user_gender_id','shop_review_positive_rate_level'][:1]
t = traindf.groupby(col)['is_trade'].mean().reset_index()
print(t.shape)
testdf = testdf[['instance_id'] + col]
test = pd.merge(testdf, t, on = col, how = 'left')
print(testdf.shape)
print(test)
prediction = test[['instance_id','is_trade']]
prediction.fillna(0,inplace=True)
prediction.columns = ['instance_id','predicted_score']
print(prediction)
prediction.to_csv('0203.txt',index = None,sep=' ')
plt.hist(prediction.predicted_score, 50, normed=1, facecolor='g', alpha=0.75)
plt.show()
