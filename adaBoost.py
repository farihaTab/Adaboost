"""
The task is to simulate an M/M/k system with a single queue.
Complete the skeleton code and produce results for three experiments.
The study is mainly to show various results of a queue against its ro parameter.
ro is defined as the ratio of arrival rate vs service rate.
For the sake of comparison, while plotting results from simulation, also produce the analytical results.
"""

import heapq
import random
# import matplotlib.pyplot as plt
import csv
import math
import sys
from numpy import genfromtxt

class Data:
    def __init__(self,csvFileName):
        self.matrix = []
        self.colCount = 0
        self.rowCount = 0
        self.csvFileName = csvFileName
        self.metaData = {}

    def readDataFromFile(self):

        lineno = 0
        with open('meta-data.txt', 'r') as f:
            for line in f:
                self.metaData.append([])
                for word in line.split():
                    self.metaData[lineno].append(word)
                lineno = lineno + 1

        csv.register_dialect('myDialect',
                             delimiter=';',
                             quoting=csv.QUOTE_ALL,
                             skipinitialspace=False)

        with open(self.csvFileName, 'r') as csvFile:
            # reader = csv.DictReader(csvFile, dialect='myDialect')
            reader = csv.reader(csvFile, dialect='myDialect')
            for row in reader:
                if self.rowCount == 0:
                    self.rowCount = len(row)
                    print('row count: ' + repr(self.rowCount))
                    continue
                self.matrix.append([])
                for i in range(self.rowCount):
                    self.matrix[self.colCount].append(row[i])
                self.colCount = self.colCount + 1
        print('col count: ' + repr(self.colCount))
        print(len(self.matrix[0]))
        print(len(self.matrix))
        csvFile.close()



class DecisionStamp:
    def __init__(self,data,weight):
        self.givenData = data
        self.weight = weight
        self.data = Data(data.csvFileName)
        data.colCount = data.colCount
        data.rowCount = data.rowCount
        data.matrix = [[] for _ in range(data.rowCount)]

    def resample(self):
        uniformStream = random.Random(101)
        myWeight = []
        total = 0
        for i in range(len(self.weight)):
            total = total + self.weight[i]
            myWeight[i] = total
        for i in range(self.data.rowCount):
            rand = uniformStream.random()*total;
            row = self.getDataRow(rand)
            for j in range(self.data.colCount):
                self.data.matrix[i][j]=row[i]

    def informationGain(self):
        None
    def learn(self):
        noOutputs = len(self.data.metaData[self.data.colCount - 1])-2   #total number  of output class
                                                                #-2 because attribute name and attribute type
        numOfEachType = [ 0 for _ in range(noOutputs)]
        for i in range(self.data.rowCount):
            output = self.data.matrix[i][self.data.colCount-1]
            for j in range(noOutputs):
                if self.data.metaData[self.data.colCount - 1][j+2]==output:
                    numOfEachType[j] = numOfEachType[j]+1
        entropy = 0#(float(total)/float(self.data.rowCount))
        # entropy = -entropy*math.log(entropy,2)
        for i in range(noOutputs):
            entropy = entropy-(float(numOfEachType[i])/float(self.data.rowCount))*math.log(float(numOfEachType[i])/float(self.data.rowCount))
        print("attribute: "+repr(self.data.metaData[self.data.colCount - 1][0])
                +" type: "+repr(self.data.metaData[self.data.colCount - 1][1])
                +" entropy: "+repr(entropy))

        infoGain = [ 0 for _ in range(self.data.colCount-1)]
        for i in range(self.data.colCount-1):# for all attribute except output

            infoGain[i] = self.calcInfoGain(i,entropy)



    def calcInfoGain(self,atrbNo, outputEntropy, noOfOutputClasses):
        if(self.data.metaData[atrbNo]=='numeric'):
            None
        else:
            outputAtrbNo = self.data.colCount-1
            noOfCats = len(self.data.metaData[atrbNo]-2)
            totalOutputsForCategories = [ [0 for x in range(noOfOutputClasses)] for _ in range(noOfCats) ]

            for i in range(self.data.rowCount):
                atrbVal = self.data.matrix[i][atrbNo]
                outputVal = self.data.matrix[i][outputAtrbNo]
                for j in range(noOfCats):
                    if self.data.metaData[atrbNo][j + 2] == atrbVal:

                for j in range(noOfCats):
                    if self.data.metaData[atrbNo][j + 2] == outputVal:
                        numOfEachType[j] = numOfEachType[j] + 1



    def getDataRow(self,val):
        for i in range(len(self.weight)):
            if(val<self.weight[i]):
                return self.givenData[i]
        return None


class AdaBoost:
    def __init__(self,data):
        self.data = data
        self.weight = [1 for _ in range(self.data.colCount)]

    def adaBoost(self):


def readDataFromFile():
    data = [[]]
    colCount = 0
    rowCount = 0
    csv.register_dialect('myDialect',
                         delimiter=';',
                         quoting=csv.QUOTE_ALL,
                         skipinitialspace=False)

    with open('bank-full.csv', 'r') as csvFile:
        # reader = csv.DictReader(csvFile, dialect='myDialect')
        reader = csv.reader(csvFile, dialect='myDialect')
        for row in reader:
            # print(len(row))
            # print(row)
            # print(dict(row))
            # print(row[2])
            if rowCount == 0:
                rowCount = len(row)
                print('row count: ' + repr(rowCount))
                continue
            data.append([])
            for i in range(rowCount):
                # print(row[i])
                data[colCount].append(row[i])
            colCount = colCount + 1
        print('col count: ' + repr(colCount))
    csvFile.close()

def main():
    # readDataFromFile()
    # data = Data('bank-full.csv')
    # data.readDataFromFile()
    # uniformStream = random.Random(101);
    # print(uniformStream.random())
    # metaDataFile = open('meta-data.txt', 'r')
    # line = metaDataFile.readline()
    # while line:


if __name__ == "__main__":
    main()
