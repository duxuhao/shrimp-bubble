import CountBubble as CB

if __name__ == "__main__":
    filename = 'B18h01m41s17jul2014y.wav'
    filenamewithlabel = '.wav'
    labelfile = '.csv'
    TrainStartTime = 3.0
    TrainEndTime = 3.5
    PredictStartTime = 3.5
    PredictEndTime = 4.0
    smoothlevel = 3
    packetlevel = 8
    windows = 2**packetlevel
    step = windows / 2
    neibour = 40 # this one looks good
    component = 2
    sample = CB.CountBubble()
    sample.GetAudio(filename, PredictStartTime, PredictEndTime)
    sample.PrepareWPE(smoothlevel, windows, step, packetlevel)
    sample.ManifoldTrain(neibour, component)
    sample.GetAudio(filename, PredictStartTime, PredictEndTime)
    sample.PrepareWPE(smoothlevel, windows, step, packetlevel)
    sample.VisualizeDimensionReduction() #visualize on other data set
    #sample.VisualizeCluster()
    #sample.GetAudio(filenamewithlabel, PredictStartTime, PredictEndTime)
    #sample.PrepareWPE(smoothlevel, windows, step, packetlevel)
    #sample.ManifoldTransform(neibour, component)
    #sample.PrepareLabelDataFrame(labelfile)
    #sample.SupervisedTrain()
    #sample.CrossValidation()
    #sample.VisualizeSupervisedLearning()