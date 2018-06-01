import heapq
import random
import csv
import bisect
import math
import sys
from numpy import genfromtxt

NUMERIC = 'numeric'
BINARY = 'binary'
CATEGORICAL = 'categorical'
seed = 101
INFINITY = float(2147483647)
MINUS_INFINITY = float(-2147483648)

class Data:
    def __init__(self, csvFileName=None, infoFileName=None, dataset=None):
        self.dataset = None
        self.matrix = []
        self.info = []
        self.totalAttributes = 0
        self.totalDatapoints = 0
        self.maxmintotal = None
        self.csvFileName = None
        self.infoFileName = None
        if dataset is None:
            self.csvFileName = csvFileName
            self.infoFileName = infoFileName
        else:
            self.dataset = dataset

    def readDataFromFile(self):
        #read meta-data / information related to data
        lineno = 0
        with open(self.infoFileName, 'r') as infoFile:
            for line in infoFile:
                self.info.append([])
                for word in line.split():
                    self.info[lineno].append(word)
                lineno = lineno + 1

        csv.register_dialect('myDialect',
                             delimiter=';',
                             quoting=csv.QUOTE_ALL,
                             skipinitialspace=False)

        #read csv file
        valNumericAtrbs = [] #max val min val total val for attributes with numeric values
        with open(self.csvFileName, 'r') as csvFile:
            reader = csv.reader(csvFile, dialect='myDialect')
            for row in reader:
                if self.totalAttributes == 0:
                    self.totalAttributes = len(row)
                    print('col count: ' + repr(self.totalAttributes))
                    continue
                self.matrix.append([])
                for i in range(self.totalAttributes):
                    # self.matrix[self.totalDatapoints].append(row[i])
                    self.matrix[self.totalDatapoints].append(self.getAtrbNo(i,row[i]))
                self.totalDatapoints = self.totalDatapoints + 1
        print('row count: ' + repr(self.totalDatapoints))
        print(len(self.matrix[0]))
        print(len(self.matrix))
        csvFile.close()

        # for atrbNo in range(self.totalAttributes):
        #     if self.info[atrbNo][1]==NUMERIC:
        #         self.info[atrbNo].append(valNumericAtrbs[atrbNo][0])
        #         self.info[atrbNo].append(valNumericAtrbs[atrbNo][1])
        #         self.info[atrbNo].append(valNumericAtrbs[atrbNo][2])
        #         # print('atrb no: '+repr(atrbNo)+' max: '+repr(valNumericAtrbs[atrbNo][0]) +' min: '+repr(valNumericAtrbs[atrbNo][1])+' total: '+repr(valNumericAtrbs[atrbNo][2]))
        # print('calculated max, min, total of numerical values')

        # print("dataset")
        # for i in range(self.totalDatapoints):
        #     for j in range(self.totalAttributes):
        #         sys.stdout.write(repr(self.matrix[i][j])+" ")
        #     print()

        # print('dataset info')
        # for atrbNo in range(self.totalAttributes):
        #     for x in range(len(self.info[atrbNo])):
        #         sys.stdout.write(repr(self.info[atrbNo][x])+' ')
        #     print()

    def getAtrbNo(self,atrbNo,atrbVal):
        if(self.info[atrbNo][1]==NUMERIC):
            val = float(atrbVal)
            return val#todo: it could be float
        totalValsOfAtrb = len(self.info[atrbNo])-2
        for i in range(totalValsOfAtrb):
            if self.info[atrbNo][i+2]==atrbVal:
                return i

    def updateNumericInfos(self,row,valNumericAtrbs):
        # valNumericAtrb calculates max min and total
        for atrbNo in range(self.totalAttributes):
            if(self.info[atrbNo][1]==NUMERIC):
                val = row[atrbNo]
                if valNumericAtrbs[atrbNo][0] < val:
                    valNumericAtrbs[atrbNo][0] = val
                if val!=-1 and val < valNumericAtrbs[atrbNo][1]:
                    valNumericAtrbs[atrbNo][1] = val
                valNumericAtrbs[atrbNo][2] = valNumericAtrbs[atrbNo][2]+val


    def getUnbiasedData(self): # todo: this is only for this dataset with binary classification
        count = 0
        negSamples = []
        posSamples = []
        valNumericAtrbs = [[0.0, INFINITY, 0.0] for _ in range(self.totalAttributes)]
        for i in range(self.totalDatapoints):
            if self.matrix[i][self.totalAttributes-1]== 0: #if output class 'YES"
                count = count + 1
                posSamples.append(self.matrix[i])
                self.updateNumericInfos(self.matrix[i],valNumericAtrbs)
            elif self.matrix[i][self.totalAttributes-1]== 1: #if outputclass 'NO'
                negSamples.append(self.matrix[i])
        print("pos len: "+repr(len(posSamples)))

        rand = random.Random(seed)
        isSampled = [False for _ in range(len(negSamples))]
        t = len(negSamples)
        i = 0
        length = len(posSamples)
        while i<length:
            idx = int(rand.random()*t)
            if isSampled[idx]==True:
                # print(idx)
                continue
            posSamples.append(negSamples[idx])
            isSampled[idx]=True
            self.updateNumericInfos(negSamples[idx], valNumericAtrbs)
            i = i+1
            # print(repr(i)+" "+repr(idx))
        self.matrix = posSamples
        self.totalDatapoints = len(self.matrix)

        self.maxmintotal = [[] for _ in range(self.totalAttributes)]
        for atrbNo in range(self.totalAttributes):
            if self.info[atrbNo][1]==NUMERIC:
                self.maxmintotal[atrbNo].append(valNumericAtrbs[atrbNo][0])
                self.maxmintotal[atrbNo].append(valNumericAtrbs[atrbNo][1])
                self.maxmintotal[atrbNo].append(valNumericAtrbs[atrbNo][2])
                # print('atrb no: '+repr(atrbNo)+' max: '+repr(valNumericAtrbs[atrbNo][0]) +' min: '+repr(valNumericAtrbs[atrbNo][1])+' total: '+repr(valNumericAtrbs[atrbNo][2]))
        print('calculated max, min, total of numerical values')

        print("len: "+repr(len(self.matrix)))


    def printInfoAboutData(self):
        print('dataset info')
        for atrbNo in range(self.totalAttributes):
            for x in range(len(self.info[atrbNo])):
                sys.stdout.write(repr(self.info[atrbNo][x]) + ' ')
            print()
        return

    def printDataset(self):
        print("dataset ")
        for i in range(self.totalDatapoints):
            for j in range(self.totalAttributes):
                sys.stdout.write(repr(self.matrix[i][j])+" ")
            print()

    def printMaxmintotal(self):
        for i in range(self.totalAttributes):
            sys.stdout.write(repr(i)+' ')
            for j in range(len(self.maxmintotal[i])):
                sys.stdout.write(repr(self.maxmintotal[i][j])+" ")
            print()

class LearningAlgorithm:
    def __init__(self,examples,weight):
        self.examples = examples
        self.weight = weight

    def hypothesis(self, sim):
        raise Exception('Unimplemented hypothesis!')


class DecisionStamp(LearningAlgorithm):
    def __init__(self,examples,weight):
        self.examples = examples #training set
        self.weight = weight
        self.trainingSet = None

    def resampleTrainingSetUsingWeights(self):
        weightSame = True
        firstWeight = self.weight[0]
        tempWeight = [0.0 for _ in range(self.examples.totalDatapoints)]
        total = 0.0
        for i in range(self.examples.totalDatapoints):
            if(self.weight != firstWeight): weightSame = False
            total = total + self.weight[i]
            tempWeight[i] = total
        # print("calculated tempweight vector. total: "+repr(total))

        if(weightSame):
            self.trainingSet = self.examples
            return

        self.trainingSet = Data(self.examples)
        self.trainingSet.totalDatapoints = self.examples.totalDatapoints
        self.trainingSet.totalAttributes = self.examples.totalAttributes
        self.trainingSet.info = self.examples.info

        # resample with replacement and create resampled training set
        uniformStream = random.Random()
        for i in range(self.trainingSet.totalDatapoints):
            rand = uniformStream.random()*float(total);
            #row = self.getDataRow(tempWeight,rand)
            rowNo = bisect.bisect_left(tempWeight,rand)
            row = self.examples.matrix[rowNo]
            self.trainingSet.matrix.append(row)
            # print(i)
        print("\tresampled training set according to weight vector")

        #print resampled set
        # for i in range(self.trainingSet.totalDatapoints):
        #     for j in range(self.trainingSet.totalAttributes):
        #         sys.stdout.write(repr(self.trainingSet.matrix[i][j])+" ")
        #     print()

    def getDataRow(self,tempWeight,val):
        for i in range(len(tempWeight)):
            if(val<tempWeight[i]):
                return self.examples.matrix[i]
        return None

    def selectRootAttribute(self): #todo: assuming numerical data has been categorized
        # calculate how many datapoint for each output class
        outputAtrbNo = self.trainingSet.totalAttributes-1
        totalOutputCategories = len(self.trainingSet.info[outputAtrbNo])-2
        outputValCounts = [0 for _ in range(totalOutputCategories)]
        for i in range(self.trainingSet.totalDatapoints):
            outputVal = self.trainingSet.matrix[i][outputAtrbNo]
            outputValCounts[outputVal] = outputValCounts[outputVal]+1

        # calculate entropy of output
        outputEntropy = 0.0
        for i in range(totalOutputCategories):
            # print(repr(i)+': count: '+repr(outputValCounts[i]))
            if outputValCounts[i]==0: continue
            outputEntropy = outputEntropy-(float(outputValCounts[i])/float(self.trainingSet.totalDatapoints))\
                              *math.log(float(outputValCounts[i])/float(self.trainingSet.totalDatapoints),2)
            # print('output entropy: ' + repr(outputEntropy))
        print('\toutput entropy: '+repr(outputEntropy))

        # calculate information gain for each input attribute
        maxInfoGain = 0
        maxInfoGainAtrbNo = 0
        valueCounts = None
        infoGains = [ 0 for _ in range(self.trainingSet.totalAttributes-1)]
        for i in range(self.trainingSet.totalAttributes-1):# for all attribute except output
            infoGains[i],valCounts = self.calcInfoGain(i,outputEntropy)
            if(infoGains[i]>maxInfoGain):
                maxInfoGain = infoGains[i]
                maxInfoGainAtrbNo = i
                valueCounts = valCounts
            # print('info gain: '+repr(infoGains[i]))
        print('\tmax info gain: '+repr(maxInfoGain)+" atrb: "+repr(maxInfoGainAtrbNo))
        # print(repr(valueCounts))
        return maxInfoGainAtrbNo,valueCounts

    def calcInfoGain(self,atrbNo, outputEntropy):#todo: assuming numerical data has been categorized
            outputAtrbNo = self.trainingSet.totalAttributes - 1
            totalOutputCategories = len(self.trainingSet.info[outputAtrbNo]) - 2
            totalCategories = len(self.trainingSet.info[atrbNo])-2
            valueCounts = [ [0 for x in range(totalOutputCategories)] for _ in range(totalCategories) ]
            catCounts = [ 0 for _ in range(totalCategories)]
            # print('atrbNo '+repr(atrbNo))
            # print('outputAtrbNo: '+repr(outputAtrbNo))
            # print('totalOutputCategories: '+repr(totalOutputCategories))
            # print('totalCategories: '+repr(totalCategories))
            # print('valueCounts: '+repr(valueCounts))
            # print('catCounts '+repr(catCounts))

            for i in range(self.trainingSet.totalDatapoints):
                catVal = self.trainingSet.matrix[i][atrbNo]
                outputVal = self.trainingSet.matrix[i][outputAtrbNo]
                # print('catval: '+repr(catVal)+" outval: "+repr(outputVal)+' '+repr(atrbNo))
                valueCounts[catVal][outputVal] = valueCounts[catVal][outputVal]+1
                catCounts[catVal] = catCounts[catVal] + 1
            # print('calculated how many datapoints in each category for attribute: '+repr(atrbNo))

            catEntropy = 0.0
            for i in range(totalCategories):
                entropy = 0.0
                for j in range(totalOutputCategories):
                    if valueCounts[i][j]==0 or catCounts[i]==0:
                        continue
                    entropy = entropy - (float(valueCounts[i][j])/float(catCounts[i]))*math.log(float(valueCounts[i][j])/float(catCounts[i]),2)
                    # print('j: '+repr(valueCounts[i][j])+' entropy: '+repr(entropy))
                catEntropy = catEntropy + (float(catCounts[i])/float(self.trainingSet.totalDatapoints))*entropy
                # print('i: '+ repr(catCounts[i])+' catEntropy: '+repr(catEntropy))

            return outputEntropy-catEntropy,valueCounts

    def learn(self):
        self.resampleTrainingSetUsingWeights()
        atrbNo, valueCounts = self.selectRootAttribute()
        outputAtrbNo = self.trainingSet.totalAttributes - 1
        totalOutputCategories = len(self.trainingSet.info[outputAtrbNo]) - 2
        hypothesis = SingleHypothesis(atrbNo, valueCounts, totalOutputCategories)
        return hypothesis

class Hypothesis:
    def __init__(self):
        print('hypothesis')

    def hypothesis(self,row):
        raise Exception('Unimplemented hypothesis!')


class SingleHypothesis(Hypothesis):
    def __init__(self,atrbNo,valueCounts,totalOutputCategories):
        self.atrbNo = atrbNo
        self.valueCounts = valueCounts
        self.totalOutputCategories = totalOutputCategories
        # self.trainingSet = trainingSet

    def hypothesis(self,row):
        atrbCat = row[self.atrbNo]
        max = 0
        maxIdx = 0
        for i in range(self.totalOutputCategories):
            # try:
            #     self.valueCounts[atrbCat][i]
            # except IndexError:
            #     print("atrbCat: "+repr(atrbCat)+" i: "+repr(i)+' atrbNo: '+repr(self.atrbNo))
            #     print(repr(self.valueCounts))
            if self.valueCounts[atrbCat][i] > max:
                max = self.valueCounts[atrbCat][i]
                maxIdx = i
        # print('classified: '+repr(maxIdx))
        return maxIdx

class AdaBoost:
    def __init__(self,trainingSet,K):
        # self.dateset = dataset
        self.K = K #the number of hypotheses in the ensemble
        self.trainingSet = trainingSet
        self.N = self.trainingSet.totalDatapoints

    def adaBoost(self):
        h = [None for _ in range(self.K)] # a vector of K hypotheses
        w = [1.0 / self.N for _ in range(self.N)] #a vector of N example weights, initially 1/N
        z = [10.0 for _ in range(self.K)] # a vector of K hypothesis weights
        outputAtrbNo = self.trainingSet.totalAttributes - 1
        totalOutputCategories = len(self.trainingSet.info[outputAtrbNo]) - 2

        for k in range(self.K):
            # h[k] ← L(examples,w)
            learningAlgo = DecisionStamp(self.trainingSet,w)
            hypothesis = learningAlgo.learn()
            h[k] = hypothesis
            # error ← 0
            # for j = 1 to N do
            #   if h[k](xj) != yj then
            #       error ← error + w[j]
            error = 0.0
            for j in range(self.N):
                if h[k].hypothesis(self.trainingSet.matrix[j]) != self.trainingSet.matrix[j][outputAtrbNo]:
                    error = error + w[j]
            print('\terr: '+repr(error))
            if error >= 0.5:
                k = k-1
                continue # if error >= 0.5 ignore this hypothesis
            # for j = 1 to N do
            #   if h[k](xj) = yj then
            #       w[j] ←w[j] · error / (1 − error)

            for j in range(self.N):
                if h[k].hypothesis(self.trainingSet.matrix[j]) == self.trainingSet.matrix[j][outputAtrbNo]:
                    w[j] = w[j]* error/(1-error)
                    # print('w['+repr(j)+']: '+repr(w[j]))

            # w← NORMALIZE(w)
            self.normalize(w)
            # z[k] ←log(1 − error) / error
            if error<0.00001:
                input('\terror < 0.0001')
            else:
                z[k] = math.log( (1-error)/error , 2)
            print('\tz['+repr(k)+']: '+repr(z[k]))
        # return WEIGHTED - MAJORITY(h, z)
        return WeightedMajorityHypothesis(h,z,totalOutputCategories)

    @staticmethod
    def normalize(w):
        total = 0.0
        for i in range(len(w)):
            total = total + w[i]
        for i in range(len(w)):
            w[i] = w[i]/total

class WeightedMajorityHypothesis(Hypothesis):

    def __init__(self,h,z,totalOutputCategories):
        self.h = h
        self.z = z
        self.totalOutputCategories = totalOutputCategories

    def hypothesis(self,row):
        K = len(self.h)
        vote = [0.0 for _ in range(self.totalOutputCategories)]
        # print(self.z)
        for k in range(K):
            hypo = self.h[k].hypothesis(row)
            vote[hypo] = vote[hypo]+ self.z[k]
            # print(repr(k)+'vote ' +repr(hypo)+' :'+ repr(vote[hypo]))

        # for i in range(self.totalOutputCategories):
        #     print('vote['+repr(i)+']: '+repr(vote[i]))

        max_vote = max(vote)
        max_index = vote.index(max_vote)
        # print('max-vote: '+repr(max_vote)+' max-idx: '+repr(max_index))
        # print('predicted class: '+repr(max_index)+' actual class: '+repr(row[len(row)-1]))
        return max_index

def getTrainingAndTestSet(dataset):
    isSampled = [False for _ in range(dataset.totalDatapoints)]  # sample from main data without replacement
    trainingSet = Data(dataset)
    trainingSet.totalDatapoints = int(0.8 * float(dataset.totalDatapoints))
    trainingSet.totalAttributes = dataset.totalAttributes
    trainingSet.info = dataset.info

    uniformStream = random.Random(seed)
    i = 0
    while i < trainingSet.totalDatapoints:
        rand = int(uniformStream.random() * dataset.totalDatapoints)
        if (isSampled[rand] == False):
            trainingSet.matrix.append(dataset.matrix[rand])
            i = i + 1
            isSampled[rand] = True
    trainingSet.totalDatapoints = len(trainingSet.matrix)
    print('seperated training set. total datapoints: '+repr(trainingSet.totalDatapoints))

    testSet = Data(dataset)
    testSet.totalDatapoints = dataset.totalDatapoints-trainingSet.totalDatapoints
    testSet.totalAttributes = dataset.totalAttributes
    testSet.info = dataset.info
    for i in range(dataset.totalDatapoints):
        if isSampled[i] == False:
            testSet.matrix.append(dataset.matrix[i])
    testSet.totalDatapoints = len(testSet.matrix)
    print('seperated test set. total datapoints: '+repr(testSet.totalDatapoints))

    return trainingSet,testSet


def divideDatesetInKparts(dataset,k):
    arrDataset = []
    isSampled = [False for _ in range(dataset.totalDatapoints)]  # sample from main data without replacement

    print('dividing dateset into k: '+repr(k)+' parts')
    for i in range(k-1):
        tempDataset = Data(dataset)
        tempDataset.totalDatapoints = dataset.totalDatapoints/k
        tempDataset.totalAttributes = dataset.totalAttributes
        tempDataset.info = dataset.info
        j = 0
        uniformStream = random.Random(seed)
        while j < tempDataset.totalDatapoints:
            rand = int(uniformStream.random() * dataset.totalDatapoints)
            if (isSampled[rand] == False):
                tempDataset.matrix.append(dataset.matrix[rand])
                j = j+1
                isSampled[rand]=True
        arrDataset.append(tempDataset)

    tempDataset = Data(dataset)
    tempDataset.totalDatapoints = dataset.totalDatapoints-(k-1)*(dataset.totalDatapoints/k)
    tempDataset.totalAttributes = dataset.totalAttributes
    tempDataset.info = dataset.info
    for i in range(dataset.totalDatapoints):
        if isSampled[i] == False:
            tempDataset.matrix.append(dataset.matrix[i])
    arrDataset.append(tempDataset)
    print('divided in '+repr(k)+' parts')

    # count = 0
    # for i in range(k):
    #     print('i: '+repr(i))
    #     print(repr(arrDataset[i].matrix))
    #     count = count+len(arrDataset[i].matrix)
    #     print('count: '+repr(count))

    return arrDataset


def classifyNumericAttributesBasedOnAvg(dataset):
    dataset.totalDatapoint = len(dataset.matrix)
    for atrbNo in range(dataset.totalAttributes):
        if(dataset.info[atrbNo][1]!=NUMERIC):
            continue
        totalVal = dataset.maxmintotal[atrbNo][2]
        avg = totalVal/dataset.totalDatapoints
        for sampleNo in range(dataset.totalDatapoints):
            val = float(dataset.matrix[sampleNo][atrbNo])
            if(val<avg):
                val = 0
            else:
                val = 1
            dataset.matrix[sampleNo][atrbNo]=int(val)
        totalAtrbs = 2
        # print('classified numeric attribute '+repr(atrbNo))

        for i in range (totalAtrbs):
            if (i+2) > len(dataset.info[atrbNo])-1:
                dataset.info[atrbNo].append(str(i))
            else:
                dataset.info[atrbNo][i + 2] = str(i)
        # print('updated info about dataset')

def checkIfNumericClassificationWasDoneProperlyOrNot(dataset):
    print("check weather numeric attributes were nicely categoried or not")
    for i in range(dataset.totalAttributes):
        if dataset.info[i][1]!=NUMERIC:
            continue
        totalcats = len(dataset.info[i])-2
        # print('atrb name: '+dataset.info[i][0])
        # print('total cats: '+repr(totalcats))
        for j in range(dataset.totalDatapoints):
            if dataset.matrix[j][i] >= totalcats:
                print('atrb name: ' + dataset.info[i][0])
                print('total cats: '+repr(totalcats))
                print(':( '+repr(dataset.matrix[j][i]))
                Exception('error in numeric attributes categorization')


def classifyNumericAttributes(dataset):
    for atrbNo in range(dataset.totalAttributes):
        if(dataset.info[atrbNo][1]!=NUMERIC):
            continue
        maxVal = dataset.maxmintotal[atrbNo][0]
        minVal = dataset.maxmintotal[atrbNo][1]
        totalAtrbs = 20
        interval = float(maxVal-minVal)/float(totalAtrbs)
        for sampleNo in range(dataset.totalDatapoints):
            x = float(dataset.matrix[sampleNo][atrbNo]) #actual numeric value
            val = int((x-minVal)/interval)
            if(val<0):
                val = totalAtrbs
                input('hihi')
            if(val>totalAtrbs):
                print('class: '+repr(val)+' '+repr(int((x-minVal)/interval)))
                Exception('what is this')
            dataset.matrix[sampleNo][atrbNo]=val
        totalAtrbs = totalAtrbs+1

        print('classified numeric attribute '+repr(atrbNo))

        for i in range (totalAtrbs):
            if (i+2) > len(dataset.info[atrbNo])-1:
                dataset.info[atrbNo].append(str(i)) #appended new val
            else:
                dataset.info[atrbNo][i + 2] = str(i) #overwrite min max

def testHypothesisAndAccuracy(h,testSet):
    # todo: only for binary classification
    testSet.totalDatapoints = len(testSet.matrix)
    error = 0
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    for i in range(testSet.totalDatapoints):
        # if h.hypothesis(testSet.matrix[i])!=testSet.matrix[i][testSet.totalAttributes-1]:
        #     error=error+1
        if h.hypothesis(testSet.matrix[i])==0 and testSet.matrix[i][testSet.totalAttributes-1]==0:
            truePos = truePos + 1
        elif h.hypothesis(testSet.matrix[i])==1 and testSet.matrix[i][testSet.totalAttributes-1]==1:
            trueNeg = trueNeg + 1
        elif h.hypothesis(testSet.matrix[i])==1 and testSet.matrix[i][testSet.totalAttributes-1]==0:
            falseNeg = falseNeg + 1
        elif h.hypothesis(testSet.matrix[i])==0 and testSet.matrix[i][testSet.totalAttributes-1]==1:
            falsePos = falsePos + 1

    error = falsePos+falseNeg
    error = (error/testSet.totalDatapoints)
    accuracy = float(truePos+trueNeg)/float(testSet.totalDatapoints)
    print('accuracy: '+repr(accuracy))
    precision = float(truePos)/float(truePos+falsePos)
    recall = float(truePos)/float(truePos+falseNeg)
    print('precision: '+repr(precision)+' recall: '+repr(recall))
    f1score = 2/(1/precision + 1/recall)
    print('f1 score: '+repr(f1score))
    return f1score, precision, recall, accuracy

def getTrainingSet(arrDataset,idx):
    # idx th dataset is test set
    # and merge other dateset into training set
    k = len(arrDataset)
    trainSet = Data(arrDataset[0])
    trainSet.totalAttributes = arrDataset[0].totalAttributes
    trainSet.info = arrDataset[0].info
    trainSet.matrix = []

    for i in range(k):
        if i==idx:
            continue
        trainSet.matrix.extend(arrDataset[i].matrix)
    trainSet.totalDatapoints = len(trainSet.matrix)
    return trainSet

def kfoldCrossValidation(dataset,K,boostingRounds):
    # todo: only for binary
    arrayOfDataset = divideDatesetInKparts(dataset, K)
    f1scoret = 0.0
    accuracyt = 0.0
    precisiont = 0.0
    recallt = 0.0
    for k in range(K):
        testSet = arrayOfDataset[k]
        testSet.totalDatapoints = len(testSet.matrix)
        trainingSet = getTrainingSet(arrayOfDataset,k)
        trainingSet.totalDatapoints = len(trainingSet.matrix)
        print('\n\tk-cross : '+repr(k))
        # print('test set'+ repr(testSet.matrix))
        # print('trainset size: '+repr(len(trainingSet.matrix)))
        # print('training set: '+repr(trainingSet.matrix))
        adaBoost = AdaBoost(trainingSet,boostingRounds)
        hypo = adaBoost.adaBoost()
        f1score, precision, recall, accuracy =  testHypothesisAndAccuracy(hypo,testSet)
        f1scoret = f1score + f1scoret
        accuracyt = accuracyt + accuracy
        precisiont = precisiont +precision
        recallt = recallt + recall
    f1scoret = f1scoret/K #avg f1 score
    accuracyt = accuracyt/K
    precisiont = precisiont/K
    recallt = recallt/K
    print('avg f1score: '+repr(f1scoret))
    print('avg accuracy: '+repr(accuracyt))
    print('avg precision: '+repr(precisiont))
    print('avg recallt: '+repr(recallt))
    return f1scoret, precisiont, recallt, accuracyt

def experiment():
    dataset = Data('bank-full.csv','meta-data.txt')
    dataset.readDataFromFile()
    dataset.getUnbiasedData()
    # classifyNumericAttributesBasedOnAvg(dataset)
    classifyNumericAttributes(dataset)
    checkIfNumericClassificationWasDoneProperlyOrNot(dataset)

    kcross = [5,10,20]
    kboost = [5,10,20,30]

    f1scorearr = [[] for _ in range(len(kboost))]
    precisionarr = [[] for _ in range(len(kboost))]
    recallarr = [[] for _ in range(len(kboost))]
    accuracyarr = [[] for _ in range(len(kboost))]
    i=0
    for kb in kboost:
        for kc in kcross:
            print()
            print('KBoost: '+repr(kb)+' KCross: '+repr(kc))
            f1score, precision, recall, accuracy =   kfoldCrossValidation(dataset,kc,kb)
            f1scorearr[i].append(f1score)
            precisionarr[i].append(precision)
            recallarr[i].append(recall)
            accuracyarr[i].append(accuracy)
        i = i+1

    for i in range(len(kboost)):
        print('kb: '+repr(kboost[i]))
        for j in range(len(kcross)):
            print('\tkc: '+repr(kcross[j])+' f1: '+repr(f1scorearr[i][j])+' prec: '+repr(precisionarr[i][j])
                  +' rec: '+repr(recallarr[i][j])+' acc: '+repr(accuracyarr[i][j]))

def kfoldCrossValidationDecisionStump(dataset,K,boostingRounds):
    # todo: only for binary
    arrayOfDataset = divideDatesetInKparts(dataset, K)
    f1scoret = 0.0
    accuracyt = 0.0
    precisiont = 0.0
    recallt = 0.0
    for k in range(K):
        testSet = arrayOfDataset[k]
        testSet.totalDatapoints = len(testSet.matrix)
        trainingSet = getTrainingSet(arrayOfDataset,k)
        trainingSet.totalDatapoints = len(trainingSet.matrix)
        print('k-cross : '+repr(k))
        # print('test set'+ repr(testSet.matrix))
        # print('trainset size: '+repr(len(trainingSet.matrix)))
        # print('training set: '+repr(trainingSet.matrix))
        N = len(trainingSet.matrix)
        w = [1.0 / N for _ in range(N)]  # a vector of N example weights, initially 1/N
        learningAlgo = DecisionStamp(trainingSet, w)
        hypo = learningAlgo.learn()
        f1score, precision, recall, accuracy =  testHypothesisAndAccuracy(hypo,testSet)
        f1scoret = f1score + f1scoret
        accuracyt = accuracyt + accuracy
        precisiont = precisiont +precision
        recallt = recallt + recall
    f1scoret = f1scoret/K #avg f1 score
    accuracyt = accuracyt/K
    precisiont = precisiont/K
    recallt = recallt/K
    print('avg f1score: '+repr(f1scoret))
    print('avg accuracy: '+repr(accuracyt))
    print('avg precision: '+repr(precisiont))
    print('avg recall: '+repr(recallt))
    return f1scoret, precisiont, recallt, accuracyt

def experimentWithDecisionStump():
    dataset = Data('bank-full.csv','meta-data.txt')
    dataset.readDataFromFile()
    dataset.getUnbiasedData()
    classifyNumericAttributesBasedOnAvg(dataset)
    checkIfNumericClassificationWasDoneProperlyOrNot(dataset)

    kcross = [5,10,20]

    f1scorearr = []
    precisionarr = []
    recallarr = []
    accuracyarr = []
    for kc in kcross:
        print()
        print('KCross: '+repr(kc))
        f1score, precision, recall, accuracy =   kfoldCrossValidation(dataset,kc,1)
        f1scorearr.append(f1score)
        precisionarr.append(precision)
        recallarr.append(recall)
        accuracyarr.append(accuracy)


    for j in range(len(kcross)):
        print('\tkc: '+repr(kcross[j])+' f1: '+repr(f1scorearr[j])+' prec: '+repr(precisionarr[j])
              +' rec: '+repr(recallarr[j])+' acc: '+repr(accuracyarr[j]))

def main():
    # experiment()
    # return
    # experimentWithDecisionStump()
    # return
    dataset = Data('bank-full.csv','meta-data.txt')
    dataset.readDataFromFile()
    dataset.getUnbiasedData()
    # dataset.printInfoAboutData()
    # dataset.printMaxmintotal()
    # classifyNumericAttributesBasedOnAvg(dataset)
    classifyNumericAttributes(dataset)
    # classifyNumericAttributesBySorting(dataset)
    # dataset.printInfoAboutData()
    # checkIfNumericClassificationWasDoneProperlyOrNot(dataset)
    kfoldCrossValidation(dataset,5,30)
    # arrayOfDataset = divideDatesetInKparts(dataset,5)
    # classifyNumericAttributes(dataset)
    # trainingSet, testSet = getTrainingAndTestSet(dataset)
    # adaBoost = AdaBoost(trainingSet,20)
    # hypo = adaBoost.adaBoost()
    # testHypothesisAndAccuracy(hypo,testSet)



if __name__ == "__main__":
    main()
