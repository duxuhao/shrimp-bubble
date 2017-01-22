import CountBubble as CB
import warnings


def GetAudioWPE(bc, filename, StartTime, EndTime, smoothlevel, windows, step, packetlevel):
    bc.GetAudio(filename, StartTime, EndTime)
    bc.PrepareWPE(smoothlevel, windows, step, packetlevel)#use wpe as feature
    return bc

def VisualizeManifold(bc, filename, StartTime, EndTime, smoothlevel, windows, step, packetlevel):
    bc = GetAudioWPE(bc, filename, 
                  StartTime, EndTime,
                  smoothlevel, windows, step, 
                  packetlevel)
    bc.VisualizeDimensionReduction()
    return bc

def TrainClaasifier(bc, filename, StartTime, EndTime, smoothlevel, windows, step, packetlevel, labelfile):
    bc = GetAudioWPE(bc, filenamewithlabel, StartTime, EndTime, smoothlevel, windows, step, packetlevel)
    bc.ManifoldTransform()
    bc.PrepareLabelDataFrame(labelfile)
    clf = bc.SupervisedTrain()
    bc.CrossValidation()
    #bc.VisualizeSupervisedLearning()
    #return clf

def PipePredict(newfile, start, end, smoothlevel, windows, step, packetlevel, preprocess, clf):
    new = CB.CountBubble()
    new = GetAudioWPE(new, newfile, start, end, smoothlevel, windows, step, packetlevel)
    return new.SupervisedPredicting(preprocess, clf)

warnings.filterwarnings("ignore")
filename = 'B18h01m41s17jul2014y.wav'
filenamewithlabel = 'B17h23m35s25apr2012y.wav'
labelfile = 'B17_Peak_Analysis.csv'
TrainStartTime = 0.0
TrainEndTime = 10.0
PredictStartTime = 0.0
PredictEndTime = 5.0
ClfStartTime = 0.0
ClfEndTime = 5.0
smoothlevel = 3
packetlevel = 8
windows = 8096
step = windows / 2
neibour = 40 # this one looks good
component = 2
SnappingShrimp = CB.CountBubble()
#SnappingShrimp.GetAudio(filenamewithlabel, PredictStartTime, PredictEndTime)
#SnappingShrimp.VisualizeTime()
SnappingShrimp = GetAudioWPE(SnappingShrimp, filename, TrainStartTime, TrainEndTime, smoothlevel, windows, step,  packetlevel)
manifold = SnappingShrimp.ManifoldTrain(neibour, component)
#VisualizeManifold(SnappingShrimp, filenamewithlabel, PredictStartTime, PredictEndTime, smoothlevel, windows, step, packetlevel)
clf = TrainClaasifier(SnappingShrimp, filenamewithlabel, ClfStartTime, ClfEndTime, smoothlevel, windows, step, packetlevel, labelfile)
#prediction = PipePredict(newfile, start, end, smoothlevel, windows, step, packetlevel, manifold, clf)
#SnappingShrimp.VisualizeCluster(2)