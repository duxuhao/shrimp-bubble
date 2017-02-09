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


def GetAudioWPE(bc, filename, StartTime, EndTime, smoothlevel, windows, step, packetlevel):
    bc.GetAudio(filename, StartTime, EndTime)
    bc.PrepareWP(smoothlevel, windows, step, packetlevel)#use wpe as feature
    return bc

def TrainClaasifier(bc, labelfile, manifold1, manifold2,clf):
    bc.ResetFeature()
    featurelist = np.zeros(9)
    bc.AddManifoldTransform(bc.WPE, manifold1);featurelist[0]=1
    #bc.AddManifoldTransform(bc.WPF, manifold2);featurelist[1]=1
    bc.AddPeak();featurelist[2]=1
    bc.AddMean();featurelist[3]=1
    #bc.AddFrequency();featurelist[4]=1
    bc.AddWPEMax();featurelist[5]=1
    bc.AddPeakEnergyRatio();featurelist[6]=1
    bc.AddMeanDeltaT();featurelist[7]=1
    bc.AddFlatness();featurelist[8]=1
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
    x,w = bc.SupervisedPredict(clf)
    for i in range(len(x)-1):
        if (x[i] + x[i+1] == 2) & (w[i] == w[i+1]):
            x[i] = 0
            w[i] = 0
    #bc.VisualizeSupervisePrediction(0)
    return x, w

def Predict(logfilename,predictfile, StartTime, EndTime, smoothlevel, windows, step, packetlevel, clf, fealist, manifoldWPEnergyModel, manifoldWPFlatnessModel):
    f = open(logfilename,'a')
    f.write('Time,Width\n')
    f.close()
    total = 0
    for i in range(int(PreStartTime), int(PreEndTime)):
        PredictSnappingShrimp = CB.CountBubble()
        PredictSnappingShrimp = GetAudioWPE(PredictSnappingShrimp, predictfile, i, i+1, smoothlevel, windows, step, packetlevel)
        Prediction,Width = ClaasifierPredict(PredictSnappingShrimp, manifoldWPEnergyModel, manifoldWPFlatnessModel, clf,fealist)
        f = open(logfilename,'a')
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
    plt.hist(df[(df.Time <= endtime) & (df.Time >= starttime)].Width,10,alpha=0.75)
    plt.xlabel('Width (ms)')
    plt.ylabel('Shrimp Quantity (tail)')
    plt.savefig(filename[:-4],dpi = 600)
    #plt.show()

warnings.filterwarnings("ignore")
pool = Pool(10)

filenamewithlabel = 'B17h23m35s25apr2012y.wav'
labelfile = 'B17_Peak_Analysis.csv'
ClfStartTime = 0.0
ClfEndTime = 146
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
record = 'Packetlevel\t{0}\tNeighbour\t{1}\tComponent\t{2}\n'.format(packetlevel, neibour, component)
print record
clf, fealist = TrainClaasifier(ClssifySnappingShrimp, labelfile, manifoldWPEnergyModel, manifoldWPFlatnessModel, clf)

## the prediction part
predictfile = 'B17h23m35s25apr2012y.wav'
PreStartTime = 0.0
PreEndTime = 60*12.0
logfilename = predictfile[:-4] + '.csv'
Predict(logfilename, predictfile,PreStartTime, PreEndTime, smoothlevel, windows, step, packetlevel, clf, fealist,manifoldWPEnergyModel, manifoldWPFlatnessModel)
hist(logfilename, 0, 147)
