import wavio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
import pandas as pd
import xgboost as xgb


class CountBubble():
    def __init__(self, filename, start, end):
        self.df = wavio.read(filename)
        self.start = start
        self.end = end
        self.data = self.df.data[int(self.start * self.df.rate) : int(self.end * self.df.rate)]
    
    def smooth(self,n):
        self.smoothdata = self.data[n:]
        for i in range(n):
            self.smoothdata = self.smoothdata + self.data[i:i-n] / (n+1)
        
    def ThresholdMethod(self, threshold = 7000, n = 1):
        self.smooth(n-1)
        a = self.smoothdata > threshold
        a = a.astype(int)
        identify = (a[1:]-a[:-1]) == 1
        print '-'*40
        print "{0} bubbles appear from {1} s to {2} s".format(np.ceil(sum(identify)[0] / 2.0), self.start, self.end)
        print '-'*40
        return identify

    def draw(self, thresshold):
        plt.plot(self.ThresholdMethod(threshold) * 100000, 'r')
        plt.plot(self.smoothdata[1:],'k')
        plt.show()
    
    def ManifoldTrain(self, target):
        train = pd.read_csv('train.csv')
        self.tf = manifold.LocallyLinearEmbedding(n_neighbors, n_components,eigen_solver='auto')
        x = self.tf.fit_transform(train.ix[:,:-3])
        if len(np.unique(train[target])) < 3:
            self.clf = xgb.XGBClassifier()
        else:
            self.clf = xgb.XGBRegressor()
        self.clf.fit(x, train[target])
    
    def ManifoldNum(self):
        df = self.data.reshape(100,-1)
        x = self.tf.transform(df)
        prediction = self.clf.predict(x)
        return sum(prediction)
    
    def ManifoldWidth(self):
        df = self.data.reshape(100,-1)
        x = self.tf.transform(df)
        prediction = self.clf.predict(x)
        return prediction

if __name__ == "__main__":
    filename = 'B18h01m41s17jul2014y.wav'
    threshold = 8000
    StartTime = 3.1
    EndTime = 3.3
    sample = CountBubble(filename, StartTime, EndTime)
    #sample.ThresholdMethod(threshold)
    #sample.draw(threshold)
    sample.ManifoldTrain(Num)
    print sample.ManifoldNum()