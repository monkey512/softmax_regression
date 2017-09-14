# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:21:14 2017

@author: Fu
"""

import csv
import numpy as np
from scipy import sparse

def toInt(array):  
    array=np.mat(array)  
    m,n=np.shape(array)  
    newArray=np.zeros((m,n))  
    for i in range(m):  
        for j in range(n):  
            newArray[i,j]=int(array[i,j])  
    return newArray  
    
def nomalizing(array):  
    m,n=np.shape(array)  
    for i in range(m):  
        for j in range(n):  
            if array[i,j]!=0:  
                array[i,j]=1  
    return array    
    
    
def loadTrainData():  
    l=[]  
    with open('train.csv') as file:  
         lines=csv.reader(file)
         i = 0
         for line in lines:
             #控制训练样本个数
             if i== 20000:
                 break
             l.append(line) #42001*785
             i = i+1
    l.remove(l[0])
    l=np.array(l)  
    label=l[:,0]
    #print(label.shape)
    data=l[:,1:] 
    #print(data.shape) 
    return nomalizing(toInt(data)),toInt(label)  

    
def loadTestData():  
    l=[]  
    with open('test.csv') as file:  
         lines=csv.reader(file) 
         #i = 0
         for line in lines: 
             #控制测试样本个数
             #if i== 100:
                 #break
             l.append(line) 
             #i = i+1
     #28001*784  
    l.remove(l[0])  
    data=np.array(l)  
    return nomalizing(toInt(data))    

def loadTestResult():  
    l=[]  
    with open('knn_benchmark.csv') as file:  
         lines=csv.reader(file)  
         for line in lines:  
             l.append(line)  
     #28001*2  
    l.remove(l[0])  
    label=np.array(l)  
    return toInt(label[:,1])      


def softmax_cost_grad(dataArray,labelArray,lamda,theta):
    #用于计算代价函数值及其梯度
    #dataArray：m*p输入矩阵，m为案例个数，p为加上常数项之后的属性个数
    #labelArray：m*1标签向量（数值型）
    #lamda：权重衰减参数weight decay parameter
    #theta：p*k系数矩阵，k为标签类别数
    #cost：总代价函数值
    #thetagrad：梯度矩阵
    #P：m*k分类概率矩阵，P（i，j）表示第i个样本被判别为第j类的概率
    #初始化m*k分类概率矩阵P
    m = dataArray.shape[0]
    #print(labelArray)
    data = np.ones(m)
    row = range(m)
    col = labelArray[0,:]
    label_extend = sparse.coo_matrix((data,(row,col)))
    #print(label_extend)
    k = label_extend.shape[1]
    P = np.zeros((m,k))
    #print(P.shape)
    #计算概率矩阵
    for i in range(m):
        test = np.exp(dataArray[i,:]*theta).getA()
        #print(test)
        #print(test/sum(sum(test)))
        P[i,:] = test/sum(sum(test))
        
    #计算代价函数值
    #print(type(theta))
    cost = -1/m*np.trace(P*label_extend.T)+lamda/2*sum(sum(np.multiply(theta,theta).getA()))
    #计算梯度
    thetagrad = -1/m*dataArray.T*(label_extend-P)+lamda*theta
    return cost,thetagrad
    
def gradAscent(dataArray,labelArray,alpha,maxCycles,lamda):
    #alpha是步长，maxCycles是迭代步数
    dataMat=np.mat(dataArray)    #size:m*n  
    #labelMat=mat(labelArray)      #size:m*1  
    m,p=np.shape(dataMat)
    #求取标签类别数
    #k = len(set(labelArray.tolist()))
    k = 10
    #随机数初始化theta矩阵
    theta = 0.005*np.random.random((p,k))
    theta = np.mat(theta)
    for i in range(maxCycles):
        print(i)
        cost_and_grad = softmax_cost_grad(dataMat,labelArray,lamda,theta)
        #cost = cost_and_grad[0]
        grad = cost_and_grad[1]
        #print(cost)
        #print(grad)
        theta=theta-alpha*grad
    return theta
    
#inX是所要测试的向量   
def classify(inX,theta):
    test = np.exp(inX*theta).getA()
    resultlist = max(test/sum(sum(test))).tolist()    
    result = resultlist.index(max(resultlist))
    return result

def saveResult(result):
    with open('result.csv','w',newline='') as myFile:      
        myWriter=csv.writer(myFile)
        writelist = []
        for i in result:
            tmp=[]
            tmp.append(i)
            writelist.append(tmp)
        print(writelist)
        myWriter.writerows(writelist)  
        
def savetheta(theta):
    theta = theta.A
    with open('theta.csv','w',newline='') as myFile:      
        myWriter=csv.writer(myFile)
        row = theta.shape[0]
        col = theta.shape[1]
        for i in range(row):
            tmp=[]
            for j in range(col):
                tmp.append(theta[i,j])
            myWriter.writerow(tmp)  

def handwritingClassTest():  
    trainData,trainLabel=loadTrainData()
    print("成功导入训练样本集")
    testData=loadTestData()
    print("成功导入测试样本集")
    testLabel=loadTestResult()  
    m,n=np.shape(testData)  
    errorCount=0  
    resultList=[]
    alpha = 0.3
    maxCycles = 500
    lamda = 0.004
    print("开始训练过程")
    theta = gradAscent(trainData,trainLabel,alpha,maxCycles,lamda)
    print("训练过程结束，开始保存theta!!!!!!!!")
    savetheta(theta)
    for i in range(m):  
         classifierResult = classify(testData[i],theta)  
         resultList.append(classifierResult)  
         print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, testLabel[0,i]))  
         if (classifierResult != testLabel[0,i]): errorCount += 1.0  
    print ("\nthe total number of errors is: %d" % errorCount) 
    print ("\nthe total error rate is: %f" % (errorCount/float(m)))  
    saveResult(resultList)
handwritingClassTest()