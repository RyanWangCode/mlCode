#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:17:08 2018

@author: oenrob
"""

import numpy as np
import matplotlib.pyplot as plt

def loadData(filename):
    dataList = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            lineData = line.strip().split('\t')
            lineFloat = list(map(float, lineData))
            dataList.append(lineFloat)
    return dataList

def calDist(vecA, vecB):
    return np.sqrt(sum(pow(vecA - vecB, 2)))

def initCenters(dataArr, k, n):
    centers = np.zeros([k, n])
    for j in range(n):
        minJ = min(dataArr[:, j])
        rangeJ = float(max(dataArr[:, j]) - minJ)
        centers[:, j] = minJ + rangeJ * np.random.rand(k)
    return centers

def kmeans(dataIn, k):
    dataArr = np.array(dataIn)
    m, n = dataArr.shape
    centers = initCenters(dataArr, k, n)
    cluster = np.zeros(m)
    cenChanged = True
    while cenChanged:
        cenChanged = False
        for i in range(m):
            minDistI = np.inf; minIdx = -1
            for j in range(k):
                dist = calDist(dataArr[i, :], centers[j, :])
                if dist < minDistI: minDistI = dist; minIdx = j;
            if minIdx != cluster[i]: cluster[i] = minIdx; cenChanged = True;
        print(centers)        
        for j in range(k):
            centers[j] = np.mean(dataArr[cluster == j], 0)

    return centers

def kMeans(dataSet, k, distMeas=calDist, createCent=initCenters):
    m, n = dataSet.shape
    clusterAssment = np.mat(np.zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k, n)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = np.inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print (centroids)
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = np.mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment[:, 0]

def plotData(dataArr, color):
    plt.scatter(dataArr[:,0], dataArr[:, 1], c=color)

dataA = loadData('testSet2.txt')
#centers = kmeans(dataA, 4)
centers, clusters = kMeans(np.array(dataA), 3)
plotData(np.array(dataA)[clusters.A.reshape(len(clusters)) == 0], 'b')
plotData(np.array(dataA)[clusters.A.reshape(len(clusters)) == 1], 'g')
plotData(np.array(dataA)[clusters.A.reshape(len(clusters)) == 2], 'r')
plotData(np.array(dataA)[clusters.A.reshape(len(clusters)) == 3], 'k')
#plt.scatter(centers[:, 0], centers[:, 1], 'r')


