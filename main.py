import CountBubble as CB
import warnings
import numpy as np
from multiprocessing import Pool
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb

def GetAudioWPE(bc, filename, StartTime, EndTime, smoothlevel, windows, step, packetlevel):
    bc.GetAudio(filename, StartTime, EndTime)
    bc.PrepareWP(smoothlevel, windows, step, packetlevel)#use wpe as feature
    return bc

def VisualizeManifold(bc, filename, StartTime, EndTime, smoothlevel, windows, step, packetlevel):
    bc = GetAudioWPE(bc, filename, 
                  StartTime, EndTime,
                  smoothlevel, windows, step, 
                  packetlevel)
    bc.VisualizeDimensionReduction()
    return bc

def TrainClaasifier(bc, labelfile, manifold1, manifold2, clf, logfilename):
    bc.ResetFeature()
    bc.AddManifoldTransform(bc.WPE, manifold1)
    bc.AddManifoldTransform(bc.WPF, manifold2)
    bc.AddWPEMax()
    bc.AddPeakEnergyRatio()
    bc.AddMeanDeltaT()
    bc.AddFlatness()
    bc.PrepareLabelDataFrame(labelfile)
    for i in xrange(1,8):
        print 'max_depth is: {0}'.format(i) 
        clf = RandomForestClassifier(max_depth=i, n_estimators=10, random_state=0)
        clf = xgb.XGBClassifier(max_depth = i)
        clf = bc.SupervisedTrain(clf)
        score = bc.CrossValidation()
        log = open(logfilename,'a')
        log.write('max_depth:\t' + str(i) +'\t'+ str(score) + '\n')
    log.write('-'*70)
    log.write('\n')
    log.close()
    #bc.VisualizeManifoldwithLabel()
    #bc.VisualizeSupervisedLearning()
    return clf

def PipePredict(newfile, start, end, smoothlevel, windows, step, packetlevel, preprocess, clf):
    new = CB.CountBubble()
    new = GetAudioWPE(new, newfile, start, end, smoothlevel, windows, step, packetlevel)
    return new.SupervisedPredicting(preprocess, clf)

warnings.filterwarnings("ignore")
pool = Pool(4)
ClassfiedList = {"Nearest Neighbors": KNeighborsClassifier(3),
                 "SVMLinear": SVC(kernel="linear", C=0.025),
                 "SVMrbf": SVC(gamma=2, C=1),
                 #"Gaussian": GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
                 "DT": DecisionTreeClassifier(max_depth=5, random_state=0),
                 "RF": RandomForestClassifier(max_depth=5, n_estimators=10, random_state=0),
                 "GBRT": GradientBoostingClassifier(random_state=0),
                 #"NeualNet": MLPClassifier(alpha=1, random_state=0),
                 "Ada": AdaBoostClassifier(),
                 "NB": GaussianNB(),
                 "xgb": xgb.XGBClassifier(n_estimators=125, max_depth = 3, learning_rate = 0.05)
                 #"QDA": QuadraticDiscriminantAnalysis()
                         }

#manifoldlist = {'LLE': manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=2,random_state=0),
                #'Spectral': manifold.SpectralEmbedding(n_components=2,n_neighbors=30,random_state=0),
                #'Iso': manifold.Isomap(n_neighbors = 30, n_components=2),
                #'MDS': manifold.MDS(n_components=2, max_iter=100),
                #'tsne': manifold.TSNE(n_components=component, init='pca', random_state=0)
                #}

clf =  ClassfiedList['RF']
#mani = manifoldlist['LLE']
filename = 'B18h01m41s17jul2014y.wav'
filenamewithlabel = 'B17h23m35s25apr2012y.wav'
labelfile = 'B17_Peak_Analysis.csv'
TrainStartTime = 0.0
TrainEndTime = 40.0
PredictStartTime = 0.0
PredictEndTime = 125.0
ClfStartTime = 0.0
ClfEndTime = 146.0
smoothlevel = 1
packetlevel = 6
windows = 2**13
step = windows / 2
logfilename = 'xgbturningsmooth2.log'
log = open(logfilename,'a')
log.write('-'*70)
log.write('\n')
log.close()
'''
for packetlevel in np.arange(3, 11):
    TrainManifoldSnappingShrimp = CB.CountBubble()
    TrainManifoldSnappingShrimp = GetAudioWPE(TrainManifoldSnappingShrimp, filename, TrainStartTime, TrainEndTime, smoothlevel, windows, step,  packetlevel)
    ClssifySnappingShrimp = CB.CountBubble()
    ClssifySnappingShrimp = GetAudioWPE(ClssifySnappingShrimp, filenamewithlabel, ClfStartTime, ClfEndTime, smoothlevel, windows, step, packetlevel)
    for neibour in np.arange(10, 20, 2): # this one looks good
        for component in np.arange(2, 10):
            log = open(logfilename,'a')
            record = 'Packetlevel\t{0}\tNeighbour\t{1}\tComponent\t{2}\n'.format(packetlevel, neibour, component)
            log.write(record)
            log.close()
            print record
            maniWPE = manifold.LocallyLinearEmbedding(n_neighbors=neibour, n_components=component,random_state=0)
            maniWPF = maniWPE
            manifoldWPEnergyModel = TrainManifoldSnappingShrimp.ManifoldTrain(TrainManifoldSnappingShrimp.WPE, maniWPE)
            manifoldWPFlatnessModel = TrainManifoldSnappingShrimp.ManifoldTrain(TrainManifoldSnappingShrimp.WPF, maniWPF)
            clf = TrainClaasifier(ClssifySnappingShrimp, labelfile, manifoldWPEnergyModel, manifoldWPFlatnessModel, clf, logfilename)

'''
SnappingShrimp = CB.CountBubble()
#SnappingShrimp.GetAudio(filenamewithlabel, PredictStartTime, PredictEndTime)
#SnappingShrimp.VisualizeTime()
SnappingShrimp = GetAudioWPE(SnappingShrimp, filename, TrainStartTime, TrainEndTime, smoothlevel, windows, step,  packetlevel)
SnappingShrimp.VisualizeWP()
manifold = SnappingShrimp.ManifoldTrain(neibour, component)
#VisualizeManifold(SnappingShrimp, filenamewithlabel, PredictStartTime, PredictEndTime, smoothlevel, windows, step, packetlevel)
clf = TrainClaasifier(SnappingShrimp, filenamewithlabel, ClfStartTime, ClfEndTime, smoothlevel, windows, step, packetlevel, labelfile,clf)
#prediction = PipePredict(newfile, start, end, smoothlevel, windows, step, packetlevel, manifold, clf)
#SnappingShrimp.VisualizeCluster(2)

