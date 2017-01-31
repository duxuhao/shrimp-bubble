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

def TrainClaasifier(bc, labelfile, manifold1, manifold2, clf, logfilename):
    bc.ResetFeature()
    bc.AddManifoldTransform(bc.WPE, manifold1)
    #bc.AddManifoldTransform(bc.WPF, manifold2)
    bc.AddPeak()
    #bc.AddFrequency()
    bc.AddWPEMax()
    bc.AddPeakEnergyRatio()
    bc.AddMeanDeltaT()
    bc.AddFlatness()
    bc.PrepareLabelDataFrame(labelfile) 
    #clf = RandomForestClassifier(max_depth=7, n_estimators=10, random_state=0)
    clf = xgb.XGBClassifier(max_depth = 3)
    #clf = MLPClassifier(hidden_layer_sizes = (i * 10,), random_state=0)
    clf = bc.SupervisedTrain(clf)
    score = bc.CrossValidation()
    bc.VisualizeClf(0)
    return clf

def ClaasifierPredict(bc, manifold1, manifold2, clf):
    bc.AddManifoldTransform(bc.WPE, manifold1)
    #bc.AddManifoldTransform(bc.WPF, manifold2)
    bc.AddPeak()
    #bc.AddFrequency()
    bc.AddWPEMax()
    bc.AddPeakEnergyRatio()
    bc.AddMeanDeltaT()
    bc.AddFlatness()
    #clf = RandomForestClassifier(max_depth=7, n_estimators=10, random_state=0)
    #clf = MLPClassifier(hidden_layer_sizes = (i * 10,), random_state=0)
    #score = bc.CrossValidation()
    x = bc.SupervisedPredict(clf)
    #bc.VisualizeSupervisePrediction(0)
    return x

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
labelfile = 'B17_Peak_Analysis.csv'
TrainStartTime = 0.0
TrainEndTime = 80.0
ClfStartTime = 0.0
ClfEndTime = 146.0
PreStartTime = 147.0
PreEndTime = 600.0
smoothlevel = 3
packetlevel = 8
windows = 2 ** 13
step = windows / 2
logfilename = 'xgbturningLLE.log'
packetlevel = 4
TrainManifoldSnappingShrimp = CB.CountBubble()
TrainManifoldSnappingShrimp = GetAudioWPE(TrainManifoldSnappingShrimp, filename, TrainStartTime, TrainEndTime, smoothlevel, windows, step,  packetlevel)
ClssifySnappingShrimp = CB.CountBubble()
ClssifySnappingShrimp = GetAudioWPE(ClssifySnappingShrimp, filenamewithlabel, ClfStartTime, ClfEndTime, smoothlevel, windows, step, packetlevel)
neibour = 35
component = 8
record = 'Packetlevel\t{0}\tNeighbour\t{1}\tComponent\t{2}\n'.format(packetlevel, neibour, component)
print record
manifoldWPEnergyModel = manifold.LocallyLinearEmbedding(n_neighbors=neibour, n_components=component,random_state=0)
#manifoldWPEnergyModel = manifold.Isomap(n_neighbors = 30, n_components=2)
manifoldWPFlatnessModel = manifoldWPEnergyModel
manifoldWPEnergyModel = TrainManifoldSnappingShrimp.ManifoldTrain(TrainManifoldSnappingShrimp.WPE, manifoldWPEnergyModel)
#manifoldWPFlatnessModel = TrainManifoldSnappingShrimp.ManifoldTrain(TrainManifoldSnappingShrimp.WPF, manifoldWPFlatnessModel)
clf = TrainClaasifier(ClssifySnappingShrimp, labelfile, manifoldWPEnergyModel, manifoldWPFlatnessModel, clf, logfilename)
for i in range(int(PreStartTime), int(PreEndTime)):
    PredictSnappingShrimp = CB.CountBubble()
    PredictSnappingShrimp = GetAudioWPE(PredictSnappingShrimp, filenamewithlabel, i, i+1, smoothlevel, windows, step, packetlevel)
    Prediction = ClaasifierPredict(PredictSnappingShrimp, manifoldWPEnergyModel, manifoldWPFlatnessModel, clf)
    count = Prediction[:]
    for j in range(1,len(Prediction)-1):
        if sum(Prediction[j-1:j+2]) > 1.5:
            count[j] = 0.5
    print '{0} to {1} s has {2} shrimps'.format(i, i+1, int(sum(count)))
