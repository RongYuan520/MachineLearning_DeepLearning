from numpy import *
import operator

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

def classify0(intX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0] #获取行数
	#距离计算
	difMat = tile(intX,(dataSetSize,1)) - dataSet#扩充成相同维度，相减
	sqDifMat = difMat ** 2#平方
	sqDistance = sqDifMat.sum(axis = 1)#横坐标相加
	distance = sqDistance ** 0.5 #开根号
	
	sortedDistIndicies = distance.argsort()#index递增排序
	#选择距离最近的k个点
	classCount = {}   # label:count
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	#排序
	sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
	return sortedClassCount[0][0]


#------------------------约会网站实例--------------------
def file2matrix(filename):
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []

	index = 0
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split("\t")
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1

	return returnMat,classLabelVector



#特征归一化

def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)

	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m,1))
	normDataSet = normDataSet/tile(ranges, (m,1))
	return normDataSet,ranges,minVals


#分类器针对约会网站的测试代码

def datingClassTest(k):
	hoRatio = 0.10
	datingDataMat,datingLabels = file2matrix("datingTestSet.txt")
	normMat,ranges,minVals = autoNorm(datingDataMat)

	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],k)
		print("pre: %d,acc: %d"%(classifierResult,datingLabels[i]))
		if(classifierResult != datingLabels[i]):
			errorCount += 1.0
	print(errorCount)
	print(numTestVecs)
	print("error rate is %f" %(errorCount/float(numTestVecs)))

#约会网站预测函数

def classifyPerson():
	resultList = ['not at all','in small doses','in largr doses']
	percentTats = float(input("playing time :"))
	ffMiles = float(input("fly miles :"))
	iceCream = float(input("numbers of iceCream :"))
	datingDataMat,datingLabels = file2matrix("datingTestSet.txt")
	normMat,ranges,minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles,percentTats,iceCream])
	classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
	print("you properly like this person:",resultList[classifierResult-1])












