import wavio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
import pandas as pd
import pywt
import matplotlib.cm as cm
import seaborn as sns
from sklearn import cluster


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
        
    def ThresholdMethod(self, threshold = 7000):
        a = self.smoothdata > threshold
        a = a.astype(int)
        identify = (a[1:]-a[:-1]) == 1
        print '-'*40
        print "{0} bubbles appear from {1} s to {2} s".format(np.ceil(sum(identify)[0] / 2.0), self.start, self.end)
        print '-'*40
        return identify
    
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
    
    def ManifoldTrain(self, neibour = 30, component = 2):
        manifoldlist = {'LLE': manifold.LocallyLinearEmbedding(n_neighbors=neibour, n_components=component,random_state=0), 
                        'Spectral': manifold.SpectralEmbedding(n_components=component,
                                n_neighbors=neibour,random_state=0),
                       'Iso': manifold.Isomap(n_neighbors = neibour, n_components=component),
                       'MDS': manifold.MDS(n_components=component, max_iter=100),
                       'tsne': manifold.TSNE(n_components=component, init='pca', random_state=0)
                        }
        mManifoldTransform = manifoldlist['LLE']
        self.manifold = mManifoldTransform.fit_transform(self.EnergyArray)
    
    def ClusterTrain(self, component = 2):
        clusterlist = {'spectral': cluster.SpectralClustering(n_clusters=component,eigen_solver='arpack',affinity="nearest_neighbors", random_state=0),
                      'Agglomerative': cluster.AgglomerativeClustering(n_clusters=component, linkage='ward'), #nice
                      'MiniBatch': cluster.MiniBatchKMeans(n_clusters=component)}
        MyCluster = clusterlist['Agglomerative']
        self.ClusterResult = MyCluster.fit_predict(self.EnergyArray)
    
    def VisualizeFrame(self,minnum,maxnum,startpoint,plt, color = 'r'):
        plt.plot(startpoint*np.array([self.step,self.step]),[minnum,maxnum],c = color, linewidth=2.0, linestyle='dashed')
        plt.plot(startpoint*np.array([self.step,self.step])+self.windows,[minnum,maxnum],c = color, linewidth=2.0, linestyle='dashed')
        plt.plot(np.array([startpoint*self.step,startpoint*self.step+self.windows]),[maxnum,maxnum],c = color, linewidth=2.0, linestyle='dashed')
        plt.plot(np.array([startpoint*self.step,startpoint*self.step+self.windows]),[minnum,minnum],c = color, linewidth=2.0, linestyle='dashed')
    
    def VisualizeThreshold(self, thresshold):
        plt.plot(self.ThresholdMethod(threshold) * 100000, 'r')
        plt.plot(self.smoothdata[1:],'k')
        plt.show() 
    
    def VisualizeWP(self):
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
    
    def VisualizeDimensionReduction(self, animation = 1):
        drawdata = self.manifold
        loop = len(drawdata)
        speed = 0.002
        startpoint = 2
        for index in range(startpoint,loop):
            plt.figure(figsize=(16,8))
            plt.subplot(121)
            plt.scatter(drawdata[:,0],drawdata[:,1], c = 'k')
            plt.scatter(drawdata[index,0],drawdata[index,1],c = 'r', s = 120)
            plt.subplot(122)
            data = self.smoothdata[self.step*(index-startpoint):(self.step*(index-startpoint)+self.windows*10)]
            plt.plot(data, c = 'k')
            minnum = min(self.smoothdata[self.step*(index):(self.step*index+self.windows)])*1.1
            maxnum = max(self.smoothdata[self.step*(index):(self.step*index+self.windows)])*1.1
            self.VisualizeFrame(minnum,maxnum,startpoint,plt)
            plt.xlim([0,self.windows * 10])
            plt.ylim([min(self.smoothdata),max(self.smoothdata)])
            plt.suptitle('Frame '+ str(index) + '/' + str(loop-1), fontsize=24)
            if animation:
                plt.ion()
                plt.pause(speed)
                plt.close()
            else:
                plt.show()
    
    def VisualizeCluster(self, animation = 1):
        loop = len(self.EnergyArray)
        speed = 0.002
        startpoint = 2
        color = ['r','b','g','y']
        for index in range(startpoint,loop):
            plt.figure(figsize=(16,8))
            data = self.smoothdata[self.step*(index-startpoint):(self.step*(index-startpoint)+self.windows*10)]
            plt.plot(data, c = 'k')
            minnum = min(self.smoothdata[self.step*(index):(self.step*index+self.windows)])*1.1
            maxnum = max(self.smoothdata[self.step*(index):(self.step*index+self.windows)])*1.1
            self.VisualizeFrame(minnum,maxnum,startpoint,plt, color[self.ClusterResult[index]])
            plt.xlim([0,self.windows * 10])
            plt.ylim([min(self.smoothdata),max(self.smoothdata)])
            plt.title('Frame '+ str(index) + '/' + str(loop-1), fontsize=24)
            if animation:
                plt.ion()
                plt.pause(speed)
                plt.close()
            else:
                plt.show()

if __name__ == "__main__":
    filename = 'B18h01m41s17jul2014y.wav'
    threshold = 8000
    StartTime = 3.0
    EndTime = 4.0
    smoothlevel = 3
    packetlevel = 8
    windows = 2**packetlevel
    step = windows / 2
    neibour = 40 # this one looks good
    component = 2
    sample = CountBubble(filename, StartTime, EndTime)
    sns.set_style("white")
    sample.smooth(smoothlevel)
    sample.CutwithWindows(windows, step)
    sample.waveletPacket(packetlevel)
    #sample.VisualizeWP()
    #sample.ThresholdMethod(threshold)
    #sample.VisualizeThreshold(threshold)
    #sample.ManifoldTrain(neibour, component)
    #sample.VisualizeDimensionReduction()
    sample.ClusterTrain()
    sample.VisualizeCluster()