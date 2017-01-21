import CountBubble as CB

def GetandPrepare(bc, filename, StartTime, EndTime, smoothlevel, windows, step, packetlevel):
    bc.GetAudio(filename, StartTime, EndTime)
    bc.PrepareWPE(smoothlevel, windows, step, packetlevel)
    return bc

def PipePredict(newfile, start, end, smoothlevel, windows, step, packetlevel, manifold, clf):
    new = CB.CountBubble()
    new = GetandPrepare(new, newfile, start, end, smoothlevel, windows, step, packetlevel)
    return new.SupervisedPredicting(manifold, clf)

filename = 'B18h01m41s17jul2014y.wav'
filenamewithlabel = '.wav'
labelfile = '.csv'
TrainStartTime = 2.0
TrainEndTime = 2.1
PredictStartTime = 4.0
PredictEndTime = 4.1
smoothlevel = 3
packetlevel = 8
windows = 2**packetlevel
step = windows / 2
neibour = 40 # this one looks good
component = 2
SnappingShrimp = CB.CountBubble()
SnappingShrimp = GetandPrepare(SnappingShrimp, filename, 
                               TrainStartTime, TrainEndTime, 
                               smoothlevel, windows, step, 
                               packetlevel)
manifold = SnappingShrimp.ManifoldTrain(neibour, component)
SnappingShrimp = GetandPrepare(SnappingShrimp, filename, 
                               PredictStartTime, PredictEndTime, 
                               smoothlevel, windows, step, 
                               packetlevel)
SnappingShrimp.VisualizeDimensionReduction() #visualize on other data set

'''
SnappingShrimp = GetandPrepare(SnappingShrimp, filenamewithlabel, 
                               ClfStartTime, ClfEndTime, 
                               smoothlevel, windows, step, 
                               packetlevel)
SnappingShrimp.ManifoldTransform()
SnappingShrimp.PrepareLabelDataFrame(labelfile)
clf = SnappingShrimp.SupervisedTrain()
SnappingShrimp.CrossValidation()
SnappingShrimp.VisualizeSupervisedLearning()
'''

#prediction = PipePredict(newfile, start, end, smoothlevel, windows, step, packetlevel, manifold, clf)
#SnappingShrimp.VisualizeCluster()