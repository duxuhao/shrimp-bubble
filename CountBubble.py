import wavio
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn import manifold
from sklearn import cluster
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score


class CountBubble():
    def __init__(self):
        self.filename = ''
        sns.set_style("white")
    
    def GetAudio(self, filename, start = 0, end = 0.1):
        """Obtain the certain time section audio data from the origin wav file.
        Parameters
        ----------
        filename: origin audio file
        start : float, start time of the wanted audio signal.
        end : float, end time of the wanted audio signal.
        """
        if self.filename != filename:
            self.filename = filename
            self.df = wavio.read(filename)
        self.start = start
        self.end = end
        self.data = self.df.data[int(self.start * self.df.rate) : int(self.end * self.df.rate)]
    
    def smooth(self,windowlength):
        """smooth the audio data for pre-processing, using a window length 
        forward average method.
        Parameters
        ----------
        windowlength: int, the length to average.
        """
        self.smoothdata = self.data[windowlength:] / float(windowlength+1)
        for i in range(windowlength):
            self.smoothdata = self.smoothdata + self.data[i:i-windowlength] / float(windowlength+1)
    
    def CutwithWindows(self, windows, step):
        """cut the smooth audio signal into different frame for processing with
        certain window length and resolution.
        Parameters
        ----------
        windows: int, the data point of each sample.
        step: the distance between each windows, must be smaller than windows and
        it decide the resolution of the sample with windows.
        """
        T = []
        self.windows = windows
        self.step = step
        try:
            for i in xrange(len(self.smoothdata) / self.step - 1):
                T.append(self.smoothdata[i*self.step:(i*self.step + self.windows)].T.tolist()[0])
            self.cutclip = pd.DataFrame(T)
        except:
            print 'Smooth data need to be done first.'
        
    def ThresholdMethod(self, threshold = 7000):
        """The simplest method for counting the shrimp with the determination
        of the threshold. Fast but not accurate
        Parameters
        ----------
        threshold: float, threshold we used, variated in different dataset and
        need to be determined manually.
        """
        a = self.smoothdata > threshold
        a = a.astype(int)
        identify = (a[1:]-a[:-1]) == 1
        print '-'*40
        print "{0} bubbles appear from {1} s to {2} s, from threshold method".format(np.ceil(sum(identify)[0] / 2.0), self.start, self.end)
        print '-'*40
        return identify
    
    def waveletPacket(self, packlevel):
        """After obtaining the frame, the Wavelet Packet Energy (WPE) feature 
        is obtain from the frame using the Wavelet Packe method.
        Parameters
        ----------
        packlevel: int, the quantity of the frequency bands of the frequency. Larger
        packlevel, higher frequency resolution and more generated features. 
        2^ packlevel must smaller than the frame data.
        """
        T = []
        for clipindex in xrange(len(self.cutclip)):
            temp = []
            wp= pywt.WaveletPacket(data=self.cutclip.ix[clipindex,:],wavelet='db1',mode='symmetric',  maxlevel = packlevel)
            for i in xrange(packlevel+1):
                for index, node in enumerate(wp.get_level(i)):
                    E = np.log(np.sqrt(np.sum(wp[node.path].data ** 2)))
                    temp.append(E)
            maxnum = float(max(temp))
            temp = list(np.array(temp) / float(max(temp))) # this function will deliminate the effect of the amplitude
            temp.append(maxnum)
            T.append(temp)
        self.WPE = np.matrix(T)
        self.WPE[self.WPE == -np.inf] = 0
        self.Feature = self.WPE
     
    def PrepareWPE(self, smoothlevel, windows, step, packetlevel):
        """Prepare the WPE from the audio data
        Parameters
        ----------
        smoothlevel: int, the length to average.
        windows: int, the data point of each sample.
        step: the distance between each windows, must be smaller than windows and
        it decide the resolution of the sample with windows.
        packlevel: int, the quantity of the frequency bands of the frequency. Larger
        packlevel, higher frequency resolution and more generated features. 
        2^ packlevel must smaller than the frame data.
        """
        print '-'*49 + '\n\tPreparing the WPE\n' + '-'*49
        self.smooth(smoothlevel)
        self.CutwithWindows(windows, step)
        self.waveletPacket(packetlevel)
        print '-'*49 + '\n\tFinish preparing the WPE\n' + '-'*49
        
    def ManifoldTrain(self, neibour = 30, component = 2,  model = 'LLE'):
        """Transfer the high dimension WPE to lower dimension using the manifold
        learning. Different methods of manifold learning can be selected as:
            1. Locally Linear Embedding; 
            2. Spectral Embedding; 
            3. Isomap; 
            4. MDS; 
            5. TSNE.
        Parameters
        ----------
        neibour: int, the quantity of distance used calculated samples.
        component: int, the dimension that convert to.
        model: string, the model you select for manifold learning
        """
        print '-'*49 + '\n\tTraining the manifold learning\n' + '-'*49
        manifoldlist = {'LLE': manifold.LocallyLinearEmbedding(n_neighbors=neibour, n_components=component,random_state=0), 
                        'Spectral': manifold.SpectralEmbedding(n_components=component,
                                n_neighbors=neibour,random_state=0),
                       'Iso': manifold.Isomap(n_neighbors = neibour, n_components=component),
                       'MDS': manifold.MDS(n_components=component, max_iter=100),
                       'tsne': manifold.TSNE(n_components=component, init='pca', random_state=0)
                        }
        self.ManifoldModel = manifoldlist[model]
        self.ManifoldModel.fit(self.Feature)
        print '-'*49 + '\n\tFinish training the manifold learning\n' + '-'*49
        return self.ManifoldModel
    
    def ManifoldTransform(self):
        """using manifold learning to transform the high dimensional features to
        low dimension using the model trained in self.ManifoldTrain()
        """
        self.ManifoldTransformData = self.ManifoldModel.transform(self.Feature)
        return self.ManifoldTransformData
    
    def PrepareLabelDataFrame(self, filename):
        """Convert the shrimp appearance time into label and make the label and 
        the origin frame into a dataframe.
        Parameters
        ----------
        filename: the file that contain the shrimp appearance time
        """
        df = pd.read_csv(filename)
        label = np.zeros((1,self.Feature.shape[0]))
        #df.time *= self.df.rate
        for i in df.Time:
            try:
                label[0][int(np.ceil(i/float(self.step))-1)] += 1
                label[0][int(np.floor(i/float(self.step))-1)] += 1
            except:
                pass
        self.LabeledDF = np.concatenate((label.T, self.ManifoldTransformData), axis=1)
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(LabeledDF[:,1:], LabeledDF[:,0], test_size=.4, random_state=0)
        s = self.LabeledDF.shape[0]
        self.X_train = self.LabeledDF[:s/2,1:]
        self.X_test = self.LabeledDF[s/2:,1:]
        self.y_train = self.LabeledDF[:s/2,0]
        self.y_test = self.LabeledDF[s/2:,0]
    
    def SupervisedTrain(self, model = 'GBRT'):
        """choose the model and use it to train the labeled data.
            1. Spectral Clustering
            2. Agglomerative Clustering
            3. MiniBatch KMeans
        Parameters
        ----------
        model: string, the model you select for supervised learning
        """
        ClassfiedList = {"Nearest Neighbors": KNeighborsClassifier(3),
                         "SVMLinear": SVC(kernel="linear", C=0.025),
                         "SVMrbf": SVC(gamma=2, C=1),
                         "Gaussian": GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
                         "DT": DecisionTreeClassifier(max_depth=3, random_state=0),
                         "RF": RandomForestClassifier(max_depth=3, n_estimators=10, max_features=1, random_state=0),
                         "GBRT": GradientBoostingClassifier(random_state=0),
                         "NeualNet": MLPClassifier(alpha=1, random_state=0),
                         "Ada": AdaBoostClassifier(),
                         "NB": GaussianNB(),
                         "QDA": QuadraticDiscriminantAnalysis()}
        self.clf =  ClassfiedList[model]
        self.clf.fit(self.X_train, self.y_train)
        return self.clf
    
    def CrossValidation(self):
        """Cross validate the model in the sample which doesn't included in
        training
        Parameters
        ----------
        component: 
        """
        self.PredictTrain = self.clf.predict_proba(self.X_train)[:,1]
        self.PredictTest = self.clf.predict_proba(self.X_test)[:,1]
        n = 49
        print '-' * n
        print '\t\t|\ttrain\t|\ttest\t|'
        print '-' * n
        print '\tAUC\t|\t'+ str(np.round(roc_auc_score(self.y_train.T, self.PredictTrain),3))+'\t|\t'+str(np.round(roc_auc_score(self.y_test.T, self.PredictTest),3))+'\t|'
        print '-' * n
        print '\tTPR\t|\t'+ str(np.round(np.sum(self.y_train * self.PredictTrain) / float(sum(self.y_train)),3))+'\t|\t'+str(np.round(np.sum(self.y_test * self.PredictTest) / float(sum(self.y_test)),3))+'\t|'
        print '-' * n
        plt.plot(self.y_test,'k')
        plt.plot(self.PredictTest,'r')
        plt.ylim([-0.1,1.1])
        plt.show()
    
    def ClusterTrain(self, component = 2, model = 'Agglomerative'):
        """Using cluster method to divide the sample into different category
        unsupervisedly. Different model can be used.
            1. Spectral Clustering
            2. Agglomerative Clustering
            3. MiniBatch KMeans
        Parameters
        ----------
        component: int, the dimension that convert to.
        model: string, the model you select for manifold learning
        """
        print '-'*49 + '\n' +'Clustering\n' + '-'*49
        clusterlist = {'spectral': cluster.SpectralClustering(n_clusters=component,eigen_solver='arpack',affinity="nearest_neighbors", random_state=0),
                      'Agglomerative': cluster.AgglomerativeClustering(n_clusters=component, linkage='ward'), #nice
                      'MiniBatch': cluster.MiniBatchKMeans(n_clusters=component)}
        MyCluster = clusterlist[model]
        return MyCluster.fit_predict(self.Feature)
    
    def SupervisedPredicting(self, preprocess1, clf):
        """A pipline to predict from raw data with the manofold model and
        classify model trained before.
        Parameters
        ----------
        manifold: model, manifold learning model.
        clf: model, classify model
        """
        return clf.predict(preprocess1.transform(self.Feature))
        
    """visualization part"""
    def VisualizeTime(self):
        """Have a brief view on the data"""
        plt.plot(self.data)
        plt.show()
        
    def VisualizationPresent(self, plt, animation, speed):
        """Present the figure plot before
        Parameters
        ----------
        plt: figure, the plot figure.
        animation: int, present in animation or statics.
        speed: float, the animination speed
        """
        if animation:
            plt.ion()
            plt.pause(speed)
            plt.close()
        else:
            plt.show()
    
    def VisualizeFrame(self,plt,minnum,maxnum,framelocation, color = 'r'):
        """Plot a frame on the signal with a window length
        Parameters
        ----------
        minnum: int, the upper bound of the frame.
        maxnum: int, the lower bound of the frame.
        step: the distance between each windows, must be smaller than windows and
        it decide the resolution of the sample with windows.
        framelocation: int, the start position of the frame
        plt: matplotlib figure, pass the figure here
        color: string, color for the frame
        """
        plt.plot(framelocation*np.array([self.step,self.step]),[minnum,maxnum],c = color, linewidth=2.0, linestyle='dashed')
        plt.plot(framelocation*np.array([self.step,self.step])+self.windows,[minnum,maxnum],c = color, linewidth=2.0, linestyle='dashed')
        plt.plot(np.array([framelocation*self.step,framelocation*self.step+self.windows]),[maxnum,maxnum],c = color, linewidth=2.0, linestyle='dashed')
        plt.plot(np.array([framelocation*self.step,framelocation*self.step+self.windows]),[minnum,minnum],c = color, linewidth=2.0, linestyle='dashed')
    
    def VisualizeThreshold(self, thresshold):
        """Visualize the result of the threshold method. Before draeing, the
        threshold method will be excuted in this function.
        ----------
        thresshold: float, threshold we used, variated in different dataset and
        need to be determined manually.
        """
        plt.plot(self.ThresholdMethod(threshold) * 100000, 'r')
        plt.plot(self.smoothdata[1:],'k')
        plt.show() 
    
    def VisualizeWP(self):
        """Visualize the WPE matrix and the corresponding rime sequence signal
        at the same time.
        """
        dimen = self.WPE.shape
        packlevel = int(np.sqrt(dimen[1]))
        Energymatrix = np.zeros([len(self.WPE), packlevel+1, 2**packlevel])
        for clipindex in xrange(dimen[0]):
            for i in xrange(dimen[0]-1):
                nodeLen = 2** (packlevel / (i+1))
                level = int(np.floor(np.log2(i+1)))
                index = i - (2 ** level - 1)
                for count in xrange(nodeLen):
                    Energymatrix[clipindex, level, count+index*nodeLen] = self.WPE[clipindex, i] * self.WPE[clipindex, -1]
        alldata = self.data
        maxnum = max(alldata)
        minnum = min(alldata)
        n = 3
        for clipindex in xrange(n,10):
            plt.figure(figsize=(16,8))
            plt.subplot(121)
            plt.plot(alldata[self.windows*(clipindex-n+1):self.windows*(clipindex+10)])
            plt.plot(np.array([self.windows,self.windows]) * (n-1),[-20000,20000],c = 'r')
            plt.plot(np.array([self.windows,self.windows]) * n,[-20000,20000],c = 'r')
            plt.ylim([maxnum,minnum])
            plt.subplot(122)
            plt.imshow(Energymatrix[clipindex,:,:], cmap=cm.jet)
            ax = plt.gca()
            ax.set_aspect('auto')
            plt.colorbar()
            plt.show()
    
    def VisualizeDimensionReduction(self, animation = 1, speed = 0.01):
        """Visualize the manifold learning result by transfering the high dimension
        data to low and visible dimension data. 
        ----------
        animation: bool, the switch of the figure presentation method. If it is
        on, the frame will continue to move forward while if it is off, the figure
        will present one by one manually.
        speed: the speed to play the animation
        """
        drawdata = self.ManifoldTransform()
        alldata = self.data
        loop = len(drawdata)
        framelocation = 3
        leng = 6
        for index in xrange(framelocation,loop):
            plt.figure(figsize=(16,8))
            plt.subplot(121)
            plt.scatter(drawdata[:,0],drawdata[:,1], c = 'k')
            plt.scatter(drawdata[index,0],drawdata[index,1],c = 'r', s = 120)
            plt.subplot(122)
            data = alldata[self.step*(index-framelocation):(self.step*(index-framelocation)+self.windows*leng)]
            plt.plot(data, c = 'k')
            minnum = min(alldata[self.step*(index):(self.step*index+self.windows)])*1.1
            maxnum = max(alldata[self.step*(index):(self.step*index+self.windows)])*1.1
            self.VisualizeFrame(plt,minnum,maxnum,framelocation)
            plt.xlim([0,self.windows * leng])
            plt.ylim([min(alldata),max(alldata)])
            plt.suptitle('Frame '+ str(index) + '/' + str(loop-1), fontsize=24)
            self.VisualizationPresent(plt, animation, speed)
    
    def VisualizeCluster(self, component = 2, animation = 1, speed = 0.01):
        """Visualize the cluster result of the data. Different categories will be
        present by different frame colors.
        ----------
        animation: bool, the switch of the figure presentation method. If it is
        on, the frame will continue to move forward while if it is off, the figure
        will present one by one manually.
        speed: the speed to play the animation
        """
        drawdata = self.ClusterTrain(component)
        alldata = self.data
        loop = len(drawdata)
        framelocation = 5
        leng = 6
        color = ['r','b','g','y','c','m']
        for index in xrange(framelocation,loop):
            plt.figure(figsize=(16,8))
            data = alldata[self.step*(index-framelocation):(self.step*(index-framelocation)+self.windows*leng)]
            plt.plot(data, c = 'k')
            minnum = min(alldata[self.step*(index):(self.step*index+self.windows)])*1.1
            maxnum = max(alldata[self.step*(index):(self.step*index+self.windows)])*1.1
            self.VisualizeFrame(plt, minnum,maxnum,framelocation, color[drawdata[index]])
            plt.xlim([0,self.windows * leng])
            plt.ylim([min(alldata),max(alldata)])
            plt.title('Frame '+ str(index) + '/' + str(loop-1), fontsize=24)
            self.VisualizationPresent(plt, animation, speed)
        
    def VisualizeSupervisedLearning(self, animation = 1, speed = 0.01):
        """Visualize the result of supervised learning, if the frame color is green,
        that means the prediction is correct. but if it is red, it means the prediction
        is wrong. The frame and precision will be update in realtime
        ----------
        animation: bool, the switch of the figure presentation method. If it is
        on, the frame will continue to move forward while if it is off, the figure
        will present one by one manually.
        speed: the speed to play the animation
        """
        alldata = self.data
        loop = len(drawdata)
        framelocation = 5
        color = ['r','g','b']
        count = np.array([0,0,0])
        result = self.y_test - self.clf.predict(X_test) + 2
        for index in xrange(framelocation,loop):
            plt.figure(figsize=(16,8))
            data = alldata[self.step*(index-framelocation):(self.step*(index-framelocation)+self.windows*10)]
            plt.plot(data, c = 'k')
            minnum = min(alldata[self.step*(index):(self.step*index+self.windows)])*1.1
            maxnum = max(alldata[self.step*(index):(self.step*index+self.windows)])*1.1
            temp = result[index]
            count[temp] += 1
            self.VisualizeFrame(plt, minnum,maxnum,framelocation, color[temp])
            plt.xlim([0,self.windows * 10])
            plt.ylim([min(alldata),max(alldata)])
            plt.title('Frame '+ str(index) + '/' + str(loop-1) + '\nresult count' + str(count[1]) + '\t' + str(count[0]) + '\t' + str(count[2]), fontsize=20)
            self.VisualizationPresent(plt, animation, speed)