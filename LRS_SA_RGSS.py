from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr
from preprocessing import preKfold
from collections import OrderedDict
import random
import numpy as np

class LRS_SA_RGSS_combination():
    def __init__(self, clf, df, RecordFolder, LossFunction, columnname, start, label,  Startcol = ['None'], PotentialAdd = [], CrossMethod = 0):
        self.clf = clf
        self.LossFunction = LossFunction
        self.df = df
        self.RecordFolder  = RecordFolder
        self.columnname = columnname
        self.TemplUsedFeatures, self.Label = start, label
        self.PotentialAdd = PotentialAdd # you need to add some potential feature here, otherwise the Simulate Anneal Arithmetic will not work
        self.Startcol = Startcol
        self.CrossMethod = CrossMethod

    def run(self):
        self.score, self.greedyscore = 1, 0
        self.remain = ''
        while (self.greedyscore < self.score) | self.ScoreUpdate(): #if the random selection have a better combination, the greedy will loop again. otherwise, the selection complete
            print('test baseline')
            self.bestscore, self.bestfeature = self.score, self.TemplUsedFeatures[:]
            self.k_fold(self.TemplUsedFeatures[:], str(0), 'baseline', coetest = 0)
            self.Greedy()
            self.ScoreUpdate()
            self.greedyscore = self.bestscore
            print('random select starts with:\n {0}\n score: {1}'.format(self.bestfeature, self.greedyscore))
#            try:
#                self.MyRandom()
#            except:
#                pass
            if self.CrossMethod != 0:
                if 1: #self.greedyscore == self.score:
                    print('small cycle cross')
                    n = 1
                    while self.ScoreUpdate() | n:
                        self.CrossTermSearch(self.bestfeature, self.bestfeature)
                        n = 0
                if self.greedyscore == self.score:
                    print('median cycle cross')
                    n = 1
                    while self.ScoreUpdate() | n:
                        self.CrossTermSearch(self.columnname, self.bestfeature)
                        n = 0
                if self.greedyscore == self.score:
                    print('large cycle cross')
                    n = 1
                    while self.ScoreUpdate() | n:
                        self.CrossTermSearch(self.columnname, self.columnname)
                        n = 0
            self.ScoreUpdate()
        print('{0}\nbest score:{1}\nbest features combination: {2}'.format('*-*' * 50, self.bestscore, self.bestfeature))

    def k_fold(self, selectcol, num, addfeature, coetest=0):#get the score with the new features list and update the best features combination
        selectcol = list(OrderedDict.fromkeys(selectcol))
        X, y = self.df, self.df[self.Label]
        totaltest = 0
        if 1:
#        try:
            for D in [24]:
                T = (X.day != D)
                X_train, X_test = X[T], X[~T]
                X_train, X_test = X_train[selectcol], X_test[selectcol]
                y_train, y_test = y[T], y[~T]
                self.clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=False,early_stopping_rounds=200)
                totaltest += self.LossFunction(y_test, self.clf.predict_proba(X_test)[:,1])
            totaltest /= 1.0
            print('Mean loss: {}'.format(totaltest))
#        except ValueError:
#            totaltest = [self.score - 1]
#        try:
        if 1:
            if np.mean(totaltest) < self.score: # only when the score improve, the program will record
                if coetest: #execute in the features adding process
                    coltemp = selectcol[:]
                    coltemp.remove(addfeature)
                    cc = [pearsonr(self.df[addfeature],self.df[ct])[0] for ct in coltemp] #check the correlation coefficient
                else:
                    cc = [0]
                if 1: #(np.abs(np.max(cc)) < 0.95): # to see the correlation coefficient between each two features
                    with open(self.RecordFolder, 'a') as f: #record all the imporved combination
                        f.write('{0}  {1}  {2}:\n{3}\t{4}\n'.format(num, addfeature, np.abs(np.max(cc)), np.round(np.mean(totaltest),6), selectcol[:], '*-' * 50))
                    self.TemplUsedFeatures, self.score = selectcol[:], np.mean(totaltest)
                    if num == 'reverse':
                        self.dele = addfeature
                    else:
                        self.remain = addfeature # updaet the performance
#        except ValueError:
#            pass

    def Greedy(self):
        col = self.columnname[:]
        print('{0}{1}{2}'.format('-' * 20, 'start greedy', '-' * 20))
        for i in self.TemplUsedFeatures:
            print(i)
            try:
                col.remove(i)
            except:
                pass
        self.dele = ''
        self.bestscore, self.bestfeature = self.score, self.TemplUsedFeatures[:]
        #print('test baseline')
        #self.k_fold(self.TemplUsedFeatures[:], str(0), 'baseline', coetest = 0)
        while (self.Startcol != self.TemplUsedFeatures) | (self.PotentialAdd != []): #stop when no improve for the last round and no potential add feature
            if self.Startcol == self.TemplUsedFeatures:
                self.ScoreUpdate()
                self.score += 0.001 #Simulate Anneal Arithmetic, step back a bit, the value need to be change
                self.TemplUsedFeatures.append(self.PotentialAdd[0])
            print('{0} {1} round {2}'.format('*' * 20, len(self.TemplUsedFeatures)+1, '*' * 20))
            if self.remain in col:
                col.remove(self.remain)
            if self.dele != '':
                col.append(self.dele)
            self.Startcol = self.TemplUsedFeatures[:]
            for sub, i in enumerate(col): #forward sequence selection add one each round
                print(i)
                print('{}/{}'.format(sub,len(col)))
                selectcol = self.Startcol[:]
                selectcol.append(i)
                self.k_fold(selectcol, str(1+sub), i, coetest = 0)
            for sr, i in enumerate(self.TemplUsedFeatures[:-1]): # backward sequence selection, -2 becuase the last 2 is just selected
                deletecol = self.TemplUsedFeatures[:] # can delete several each round
                if i in deletecol:
                    deletecol.remove(i)
                print(i)
                print('reverse {}/{}'.format(sr,len(self.TemplUsedFeatures[:-1])))
                self.k_fold(deletecol, 'reverse', i, coetest = 0)
            for i in self.TemplUsedFeatures:
                if i in self.PotentialAdd:
                    self.PotentialAdd.remove(i)
        print('{0}{1}{2}'.format('-' * 20, 'complete greedy', '-' * 20))

    def MyRandom(self):
        self.ScoreUpdate()
        col = self.columnname[:]
        print('{0}{1}{2}'.format('-' * 20, 'start random', '-' * 20))
        for i in self.bestfeature:
            col.remove(i)
        for t in range(4,8): # add 4 to 8 features randomly
            print('add {} features'.format(t))
            for i in range(50): # run 50 round each quantity
                selectcol = random.sample(col, t)
                for add in self.bestfeature:
                    selectcol.append(add)
                self.k_fold(selectcol, str(i), 'None')
        print('{0}{1}{2}'.format('-' * 20, 'complete random', '-' * 20))

    def ScoreUpdate(self):
        if self.score == 1:
            return True
        elif self.score < self.bestscore:
            self.bestscore, self.bestfeature = self.score, self.TemplUsedFeatures[:]
            return True
        return False

    def CrossTermSearch(self, col1, col2):
        self.ScoreUpdate()
        Effective = []
        for c1 in col1:
            for c2 in col2[::-1]:
                for oper in self.CrossMethod.keys():
                    newcolname = c1+oper+c2
                    self.df[newcolname] = self.CrossMethod[oper](self.df[c1], self.df[c2])
                    selectcol = self.bestfeature[:]
                    selectcol.append(newcolname)
                    try:
                        self.k_fold(selectcol, 'cross term', newcolname, coetest = 0)
                    except:
                        pass
                    if self.ScoreUpdate():
                        Effective.append(newcolname)
                    else:
                        self.df.drop(newcolname, axis = 1,inplace=True)
        Effective.remove(self.remain)
        for rm in Effective:
             self.df.drop(rm, axis = 1, inplace=True)
        self.columnname.append(self.remain)

