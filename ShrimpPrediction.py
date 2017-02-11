import time
import matplotlib.pyplot as plt
import pandas as pd
import CountBubble as CB
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


def GetAudioWPE(bc, filename, StartTime, EndTime, smoothlevel, windows, step, packetlevel,averagelevel=0):
    bc.GetAudio(filename, StartTime, EndTime)
    if averagelevel:
        bc.data = bc.data * averagelevel / np.sqrt(np.mean(bc.data **2))
    bc.PrepareWP(smoothlevel, windows, step, packetlevel)#use wpe as feature
    return bc

def TrainClaasifier(bc, labelfile, manifold1, manifold2):
    bc.ResetFeature()
    featurelist = np.zeros(100)
    bc.AddManifoldTransform(bc.WPE, manifold1);featurelist[0]=1
    #bc.AddManifoldTransform(bc.WPF, manifold2);featurelist[1]=1
    bc.AddPeak();featurelist[2]=1
    bc.AddMean();featurelist[3]=1
    #bc.AddFrequency();featurelist[4]=1
    bc.AddWPEMax();featurelist[5]=1
    bc.AddPeakEnergyRatio();featurelist[6]=1
    bc.AddMeanDeltaT();featurelist[7]=1
    bc.AddFlatness();featurelist[8]=1
    #bc.AddDTW();featurelist[9]=1
    bc.PrepareLabelDataFrame(labelfile)
    clf = xgb.XGBClassifier(max_depth = 4)
    clf = bc.SupervisedTrain(clf)
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

def Predict(logfilename,predictfile, StartTime, EndTime, smoothlevel, windows, step, packetlevel, clf, fealist, manifoldWPEnergyModel, manifoldWPFlatnessModel, averagelevel = 0):
    f = open('output/'+logfilename,'a')
    f.write('Time,Width\n')
    f.close()
    total = 0
    for i in range(int(PreStartTime), int(PreEndTime)):
        PredictSnappingShrimp = CB.CountBubble()
        PredictSnappingShrimp = GetAudioWPE(PredictSnappingShrimp, predictfile, i, i+1, smoothlevel, windows, step, packetlevel, averagelevel)
        Prediction,Width = ClaasifierPredict(PredictSnappingShrimp, manifoldWPEnergyModel, manifoldWPFlatnessModel, clf,fealist)
        f = open('output/'+logfilename,'a')
        presicetime = np.where(Prediction == 1)
        for t,w in enumerate(Width[presicetime]):
            f.write('{0},{1}\n'.format(np.round(i+float(presicetime[0][t]-0.5)/len(Prediction),3), w))
        f.close()
        total += 1
        #print '{0}\t'.format(total),


def hist(filename, starttime, endtime, thres = 10):
    df = pd.read_csv('output/'+filename)
    df = df[df.Width > thres]
    df.Width /= 0.192
    plt.hist(df[(df.Time <= endtime) & (df.Time >= starttime)].Width,10,alpha=0.75)
    plt.xlabel('Width (ms)')
    plt.ylabel('Shrimp Quantity (tail)')
    plt.savefig('output/'+filename[:-4],dpi = 600)
    plt.close()

warnings.filterwarnings("ignore")
pool = Pool(12)
smoothlevel = 1
windows = 2 ** 13
step = windows / 2
packetlevel, neibour, component = 7, 35, 7

## train the manifold model
filename = 'B18h01m41s17jul2014y.wav'
TrainStartTime = 0.0
TrainEndTime = 80.0
TrainManifoldSnappingShrimp = CB.CountBubble()
TrainManifoldSnappingShrimp = GetAudioWPE(TrainManifoldSnappingShrimp, filename, TrainStartTime, TrainEndTime, smoothlevel, windows, step,  packetlevel)
manifoldWPEnergyModel = manifold.LocallyLinearEmbedding(n_neighbors=neibour, n_components=component,random_state=0)
manifoldWPFlatnessModel = manifoldWPEnergyModel
manifoldWPEnergyModel = TrainManifoldSnappingShrimp.ManifoldTrain(TrainManifoldSnappingShrimp.WPE, manifoldWPEnergyModel)
#manifoldWPFlatnessModel = TrainManifoldSnappingShrimp.ManifoldTrain(TrainManifoldSnappingShrimp.WPF, manifoldWPFlatnessModel)

##train the classification model
filenamewithlabel = 'B17h23m35s25apr2012y.wav'
labelfile = 'B17_Peak_Analysis.csv'
ClfStartTime = 0.0
ClfEndTime = 146
ClssifySnappingShrimp = CB.CountBubble()
ClssifySnappingShrimp = GetAudioWPE(ClssifySnappingShrimp, filenamewithlabel, ClfStartTime, ClfEndTime, smoothlevel, windows, step, packetlevel)
ClfMean = np.sqrt(np.mean(ClssifySnappingShrimp.data **2))
record = 'Packetlevel\t{0}\tNeighbour\t{1}\tComponent\t{2}\n'.format(packetlevel, neibour, component)
print record
clf, fealist = TrainClaasifier(ClssifySnappingShrimp, labelfile, manifoldWPEnergyModel, manifoldWPFlatnessModel)

## the prediction part


predictionfilelist = ['B09h39m21s17jul2011y.wav','B11h08m25s24aug2007y.wav','B12h31m11s04oct2007y.wav','B12h35m21s29apr2008y.wav','B16h03m56s10sep2007y.wav','B17h12m11s09jul2009y.wav','B18h17m19s19jan2009y.wav','B18h39m48s26apr2012y.wav','B18h01m41s17jul2014y.wav','B17h23m35s25apr2012y.wav','B12h08m04s29apr2008y.wav']
for predictfile in predictionfilelist[:]:
    start = time.time()
    PreStartTime = 0.0
    PreEndTime = 200000
    logfilename = predictfile[:-4] + '.csv'
    try:
        try:
            Predict(logfilename, predictfile, PreStartTime, PreEndTime, smoothlevel, windows, step, packetlevel, clf, fealist,manifoldWPEnergyModel, manifoldWPFlatnessModel, averagelevel = ClfMean)
        except:
            hist(logfilename, PreStartTime, PreEndTime)
    except:
        pass
    print predictfile,
    print 'used time consumption: {0}'.format(time.time() - start)
