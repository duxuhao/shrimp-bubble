import pandas as pd
import CountBubble_Normalized as CB
import warnings
import numpy as np
from multiprocessing import Pool
from sklearn import manifold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


def GetAudioWPE(bc, filename, StartTime, EndTime, smoothlevel, windows, step, packetlevel):
    bc.GetAudio(filename, StartTime, EndTime)
    bc.PrepareWP(smoothlevel, windows, step, packetlevel, StartTime, EndTime)#use wpe as feature
    return bc

def TrainClaasifier(bc, labelfile, manifold1, manifold2,clf):
    bc.ResetFeature()
    featurelist = np.zeros(100)
    bc.AddManifoldTransform(bc.WPE, manifold1);featurelist[0]=1;print bc.Feature.shape
    #bc.AddManifoldTransform(bc.WPF, manifold2);featurelist[1]=1
    #bc.AddPeak();featurelist[2]=1
    #bc.AddMean();featurelist[3]=1
    bc.AddFrequency();featurelist[4]=1;print bc.Feature.shape
    bc.AddWPEMax();featurelist[5]=1;print bc.Feature.shape
    #bc.AddPeakEnergyRatio();featurelist[6]=1
    #bc.AddMeanDeltaT();featurelist[7]=1
    #bc.AddFlatness();featurelist[8]=1
    bc.AddDTW();featurelist[9]=1;print bc.Feature.shape
    bc.PrepareLabelDataFrame(labelfile)
    clf = xgb.XGBClassifier(max_depth = 8)
    clf = bc.SupervisedTrain(clf)
    print clf.feature_importances_
    score = bc.CrossValidation()
    #bc.VisualizeClf(0)
    return clf, featurelist

def ClaasifierPredict(bc, manifold1, manifold2, clf, featurelist):
    if featurelist[0]:
        bc.AddManifoldTransform(bc.WPE, manifold1)
    if featurelist[1]:
        bc.AddManifoldTransform(bc.WPF, manifold2)
    if featurelist[2]:
        bc.AddPeak()
    if featurelist[3]:
        bc.AddMean()
    if featurelist[4]:
        bc.AddFrequency()
    if featurelist[5]:
        bc.AddWPEMax()
    if featurelist[6]:
        bc.AddPeakEnergyRatio()
    if featurelist[7]:
        bc.AddMeanDeltaT()
    if featurelist[8]:
        bc.AddFlatness()
    if featurelist[9]:
        bc.AddDTW()
    x,w = bc.SupervisedPredict(clf)
    for i in range(len(x)-1):
        if (x[i] + x[i+1] == 2) & (w[i] == w[i+1]):
            x[i] = 0
            w[i] = 0
    #bc.VisualizeSupervisePrediction(0)
    return x, w

def Predict(dir,logfilename, predictfile, StartTime, EndTime, smoothlevel, windows, step, packetlevel, clf, fealist, manifoldWPEnergyModel, manifoldWPFlatnessModel):
    f = open(dir + '/' + logfilename,'a')
    f.write('Time,Width\n')
    f.close()
    total = 0
    PredictSnappingShrimp = CB.CountBubble()
    PredictSnappingShrimp.GetAudio(predictfile, StartTime, EndTime)
    for i in range(int(StartTime), int(EndTime)):
        PredictSnappingShrimp.PrepareWP(smoothlevel, windows, step, packetlevel, i, i+1)
        #PredictSnappingShrimp = CB.CountBubble()
        #PredictSnappingShrimp = GetAudioWPE(PredictSnappingShrimp, predictfile, i, i+1, smoothlevel, windows, step, packetlevel)
        Prediction,Width = ClaasifierPredict(PredictSnappingShrimp, manifoldWPEnergyModel, manifoldWPFlatnessModel, clf,fealist)
        f = open(dir + '/' + logfilename,'a')
        presicetime = np.where(Prediction == 1)
        for t,w in enumerate(Width[presicetime]):
            f.write('{0},{1}\n'.format(np.round(i+float(presicetime[0][t]-0.5)/len(Prediction),3), w))
        f.close()
        total += 1
        print '{0}\t'.format(total),


def hist(filename, starttime, endtime, thres = 15):
    df = pd.read_csv(filename)
    df = df[df.Width > thres]
    df.Width /= 192
    plt.hist(df[(df.Time <= endtime) & (df.Time >= starttime)].Width,10,normed=1,alpha=0.75)
    plt.xlabel('Width (ms)')
    plt.ylabel('percentage')
    plt.show()

warnings.filterwarnings("ignore")
pool = Pool(10)

ClassfiedList = {"Nearest Neighbors": KNeighborsClassifier(3),
                 "SVMLinear": SVC(kernel="linear", C=0.025),
                 "SVMrbf": SVC(gamma=2, C=1),
                 "Gaussian": GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
                 "DT": DecisionTreeClassifier(max_depth=5, random_state=0),
                 "RF": RandomForestClassifier(max_depth=5, n_estimators=10, random_state=0),
                 "GBRT": GradientBoostingClassifier(random_state=0),
                 "NeualNet": MLPClassifier(alpha=1, random_state=0),
                 "Ada": AdaBoostClassifier(),
                 "NB": GaussianNB(),
                 "xgb": xgb.XGBClassifier(n_estimators=125, max_depth = 3, learning_rate = 0.05),
                 "QDA": QuadraticDiscriminantAnalysis()
                         }

manifoldlist = {'LLE': manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=2,random_state=0),
                'Spectral': manifold.SpectralEmbedding(n_components=2,n_neighbors=30,random_state=0),
                'Iso': manifold.Isomap(n_neighbors = 30, n_components=2),
                'MDS': manifold.MDS(n_components=2, max_iter=100),
                'tsne': manifold.TSNE(n_components=2, init='pca', random_state=0)
                }

clf =  ClassfiedList['RF']
mani = manifoldlist['LLE']
filename = 'B18h01m41s17jul2014y.wav'
filenamewithlabel = 'B17h23m35s25apr2012y.wav'
#labelfile = 'B17_Peak_Analysis.csv'
labelfile = 'sginal.csv'
TrainStartTime = 0.0
TrainEndTime = 80.0
ClfStartTime = 0.0
#ClfEndTime = 146.0
ClfEndTime = 166
PreStartTime = 0.0
PreEndTime = 60*12.0
smoothlevel = 1
windows = 2 ** 13
step = windows / 2
packetlevel, neibour, component = 4,44,7 #7, 30, 9
TrainManifoldSnappingShrimp = CB.CountBubble()
TrainManifoldSnappingShrimp = GetAudioWPE(TrainManifoldSnappingShrimp, filename, TrainStartTime, TrainEndTime, smoothlevel, windows, step,  packetlevel)
ClssifySnappingShrimp = CB.CountBubble()
ClssifySnappingShrimp = GetAudioWPE(ClssifySnappingShrimp, filenamewithlabel, ClfStartTime, ClfEndTime, smoothlevel, windows, step, packetlevel)
record = 'Packetlevel\t{0}\tNeighbour\t{1}\tComponent\t{2}\n'.format(packetlevel, neibour, component)
print record
manifoldWPEnergyModel = manifold.LocallyLinearEmbedding(n_neighbors=neibour, n_components=component,random_state=0)
manifoldWPFlatnessModel = manifoldWPEnergyModel
manifoldWPEnergyModel = TrainManifoldSnappingShrimp.ManifoldTrain(TrainManifoldSnappingShrimp.WPE, manifoldWPEnergyModel)
#manifoldWPFlatnessModel = TrainManifoldSnappingShrimp.ManifoldTrain(TrainManifoldSnappingShrimp.WPF, manifoldWPFlatnessModel)
clf, fealist = TrainClaasifier(ClssifySnappingShrimp, labelfile, manifoldWPEnergyModel, manifoldWPFlatnessModel, clf)
'''
dir = 'test_set'
predictionfilelist = ['B11h08m25s24aug2007y.wav', 'B16h03m56ssep2007y.wav', 'B18h17m19s19jan2009y.wav','B09h39m21s17jul2011y.wav','B12h35m21s29apr2008y.wav','B17h12m11s09jul2009y.wav','B17h23m35s25apr2012','B12h31m11s04oct2007y.wav']#['B17h23m35s25apr2012y.wav','B09h39m21s17jul2011y.wav','B17h12m11s09jul2009y.wav','B18h17m19s19jan2009y.wav','B18h39m48s26apr2012y.wav','B18h01m41s17jul2014y.wav','B12h31m11s04oct2007y.wav','B11h08m25s24aug2007y.wav','B12h35m21s29apr2008y.wav','B16h03m56s10sep2007y.wav','B12h08m04s29apr2008y.wav']
for predictfile in predictionfilelist[:]:
    logfilename = predictfile[:-4]+'formal.csv'
    PreStartTime = 0.0
    PreEndTime = 60*100.0
    try:
        Predict(dir, logfilename,predictfile, PreStartTime, PreEndTime, smoothlevel, windows, step, packetlevel, clf, fealist,manifoldWPEnergyModel, manifoldWPFlatnessModel)
    except:
        pass
'''
#hist(logfilename, 0, 147)

