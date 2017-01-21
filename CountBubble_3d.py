import wavio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
import pandas as pd
import pywt
import matplotlib.cm as cm
import seaborn as sns


class CountBubble():
    def __init__(self, filename, start, end):
        self.df = wavio.read(filename)
        self.start = start
        self.end = end
        self.data = self.df.data[int(self.start * self.df.rate) : int(self.end * self.df.rate)]
    
    def smooth(self,n):
        self.smoothdata = self.data[n:] / float(n+1)
        for i in range(n):
            self.smoothdata = self.smoothdata + self.data[i:i-n] / float(n+1)
        
    def ThresholdMethod(self, threshold = 7000, n = 4):
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
    
    def CutwithWindows(self, windows, step):
        T = []
        self.windows = windows
        self.step = step
        try:
            for i in xrange(len(self.smoothdata) / self.step - 1):
                T.append(self.smoothdata[i*self.step:(i*self.step + self.windows)].T.tolist()[0])
            self.cutclip = pd.DataFrame(T)
        except:
            print 'Smooth data need to be done first.'
    
    def waveletPacket(self, packlevel):
        self.Energe = np.zeros([len(self.cutclip), packlevel+1, 2**packlevel])
        T = []
        for clipindex in range(len(self.cutclip)):
            temp = []
            wp= pywt.WaveletPacket(data=self.cutclip.ix[clipindex,:],wavelet='db1',mode='symmetric',  maxlevel = packlevel)
            for i in xrange(packlevel+1):
                for index, node in enumerate(wp.get_level(i)):
                    E = np.log(np.sqrt(np.sum(wp[node.path].data ** 2)))
                    nodeLen = 2** (packlevel / (i+1))
                    temp.append(E)
                    for count in xrange(nodeLen):
                        self.Energe[clipindex, i, count+index*nodeLen] = E
            maxnum = float(max(temp))
            temp = list(np.array(temp) / float(max(temp))) # this function will deliminate the effect of the amplitude
            temp.append(maxnum)
            T.append(temp)
        self.EnergyArray = np.matrix(T)
    
    def draw_wp(self):
        maxnum = max(self.smoothdata)
        minnum = min(self.smoothdata)
        n = 3
        for clipindex in range(n,10):
            plt.figure(figsize=(16,8))
            plt.subplot(121)
            plt.plot(self.smoothdata[self.windows*(clipindex-n+1):self.windows*(clipindex+10)])
            plt.plot(np.array([self.windows,self.windows]) * (n-1),[-20000,20000],c = 'r')
            plt.plot(np.array([self.windows,self.windows]) * n,[-20000,20000],c = 'r')
            plt.ylim([maxnum,minnum])
            plt.subplot(122)
            plt.imshow(self.Energe[clipindex,:,:], cmap=cm.jet)
            ax = plt.gca()
            ax.set_aspect('auto')
            plt.colorbar()
            plt.show()
    
    def ManifoldTrain(self, neibour = 30, component = 2):
        self.tf = manifold.LocallyLinearEmbedding(neibour, component,eigen_solver='auto',random_state=0)
        self.x = self.tf.fit_transform(self.EnergyArray[:,1:])
        
    def plotManifoldTrain(self):
        loop = len(self.x)
        #loop = 2
        #loop = 50
        speed = 0.01
        for index in range(1,loop):
            plt.figure(figsize=(16,8))
            ax = plt.subplot(121, projection='3d')
            ax.scatter(self.x[:,0],self.x[:,1],self.x[:,2], c = 'k')
            ax.scatter(self.x[index,0],self.x[index,1],self.x[index,2],c = 'r',s = 100)
            plt.subplot(122)
            data = self.smoothdata[self.step*(index-1):(self.step*(index-1)+self.windows*10)]
            plt.plot(data, c = 'k')
            minnum = min(self.smoothdata[self.step*(index):(self.step*index+self.windows)])*1.1
            maxnum = max(self.smoothdata[self.step*(index):(self.step*index+self.windows)])*1.1
            plt.plot(np.array([self.step,self.step]),[minnum,maxnum],c = 'r', linewidth=2.0, linestyle='dashed')
            plt.plot(np.array(np.array([self.step,self.step])+self.windows),[minnum,maxnum],c = 'r', linewidth=2.0, linestyle='dashed')
            plt.plot(np.array([self.step,self.step+self.windows]),[maxnum,maxnum],c = 'r', linewidth=2.0, linestyle='dashed')
            plt.plot(np.array([self.step,self.step+self.windows]),[minnum,minnum],c = 'r', linewidth=2.0, linestyle='dashed')
            plt.xlim([0,self.windows * 10])
            plt.suptitle('Frame '+ str(index) + '/' + str(loop-1), fontsize=24)
            #plt.ion()
            #plt.pause(speed)
            #plt.close()
            plt.show()

if __name__ == "__main__":
    filename = 'B18h01m41s17jul2014y.wav'
    threshold = 8000
    StartTime = 4.2
    EndTime = 4.4
    packetlevel = 8
    windows = 2**packetlevel
    #windows = 256
    step = windows / 2
    neibour = 40 # this one looks good
    component = 3
    sample = CountBubble(filename, StartTime, EndTime)
    sns.set_style("white")
    sample.smooth(3)
    sample.CutwithWindows(windows, step)
    sample.waveletPacket(packetlevel)
    #sample.draw_wp()
    sample.ThresholdMethod(threshold)
    #sample.draw(threshold)
    sample.ManifoldTrain(neibour, component)
    sample.plotManifoldTrain()
