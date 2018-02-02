# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

def loadData(filename):
	dataArr = []; labelVec = []
	f = open(filename, 'r')
	for line in f.readlines():
		lineArr = []
		curline = line.strip().split('\t')
		for i in range(len(curline)-1):
			lineArr.append(float(curline[i]))
		dataArr.append(lineArr)
		labelVec.append(float(curline[-1]))
	f.close()
	return dataArr, labelVec

def standReg(XArr, yList):
    XMat = np.mat(XArr); yVec = np.mat(yList).T
    XTX = XMat.T * XMat
    if np.linalg.det(XTX) == 0.0:
        print('This matrix is singular')
        return
    wgt = XTX.I * (XMat.T * yVec)
    return wgt

def wgtGaussReg(XArr, yList, testPoint, k = 1.0):
    XMat = np.mat(XArr); yVec = np.mat(yList).T
    m = XMat.shape[0]
    wgtMat = np.mat(np.eye(m))
    for i in range(m):
        wgtMat[i, i] = np.exp((testPoint - XMat[i, :]) * (testPoint - XMat[i, :]).T / (-2.0 * k ** 2))
    XTWX = XMat.T * (wgtMat * XMat)
    if np.linalg.det(XTWX) == 0.0: return
    wgt = XTWX.I * (XMat.T * wgtMat * yVec)
    return testPoint * wgt

def wgtGaussTest(testArr, XArr, yList, k = 1.0):
    m = testArr.shape[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = wgtGaussReg(XArr, yList, testArr[i], k)
    return yHat


