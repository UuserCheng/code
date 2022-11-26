from __future__ import division
import sys, os
import random
import numpy
import copy
import operator
import time
import threading
import math
# import cPickle
import heapq
import itertools
import random
import bisect


class Point:
    def __init__(self, p, dim, id=-1):
        self.coordinates = []
        self.pointList = []
        self.id = id
        self.pointCentroid = 0
        for x in range(0, dim):
            self.coordinates.append(p[x])
        # 	这个属性是干嘛的?
        self.centroid = None


class Centroid:
    count = 0

    def __init__(self, point):
        self.point = point
        self.count = Centroid.count
        self.pointList = []
        self.centerPos = []
        self.predictions = []
        self.centerPos.append(self.point)
        self.centroid = None
        Centroid.count += 1

    def update(self, point):
        self.point = point
        self.centerPos.append(self.point)

    def addPoint(self, point):
        self.pointList.append(point)

    def removePoint(self, point):
        self.pointList.remove(point)


class Kmeans:
    # 多加了一个参数centroidsToRemember，下面有几行变了
    def __init__(self, k, pointList, kmeansThreshold, centroidsToRemember, initialCentroids=None):
        self.pointList = []
        self.numPoints = len(pointList)
        self.k = k
        self.initPointList = []
        # 是不是0啊？因为没有传参数进来==不是，差个参数就是不能运行的，所以应该是多少？
        self.centroidsToRemember = int(k * centroidsToRemember / 100)
        print("Centroids to Remember:", self.centroidsToRemember)
        self.dim = len(pointList[0])
        self.kmeansThreshold = kmeansThreshold
        self.error = None
        self.errorList = []
        # 字典形式，新加的，距离其最近的几个
        self.closestClusterDistance = {}
        self.centroidDistance = {}
        i = 0
        for point in pointList:
            p = Point(point, self.dim, i)
            i += 1
            self.pointList.append(p)
            # 完成点与最近cluster的初始化
            self.closestClusterDistance[p.id] = -1
            self.centroidDistance[p.id] = []

        # seeds和selectSeeds是在这用的
        if initialCentroids != None:
            self.centroidList = self.seeds(initialCentroids)
        else:
            self.centroidList = self.selectSeeds(self.k)
        self.mainFunction()

    # 选择种子的算法没看懂？==好像就是随机选取k个
    def selectSeeds(self, k):
        seeds = random.sample(self.pointList, k)
        centroidList = []
        for seed in seeds:
            # 这是进行了一个强制的类型转换嘛？==对滴Point-->Centroid
            centroidList.append(Centroid(seed))
        return centroidList

    def seeds(self, initList):
        centroidList = []
        for seed in initList:
            centroidList.append(Centroid(seed))
        return centroidList

    def getDistance(self, point1, point2):
        distance = 0
        for x in range(0, self.dim):
            distance += (point1.coordinates[x] - point2.coordinates[x]) ** 2
        return (distance) ** (0.5)

    # 完成了remember一系列的填充，只会有指定数量的存下来，但好像没有措施保证是最短的几个点？？
    def getCentroidInit(self, point):
        minDist = -1
        pos = 0
        # 以一个点出发，判断其与目前所有中心点的d
        for centroid in self.centroidList:
            # 获得当前点与中心各点的距离，循环来
            dist = self.getDistance(point, centroid.point)
            # 如果长度小于需要记住的，不管，直接插入
            if len(self.centroidDistance[point.id]) < self.centroidsToRemember:
                # bisec.insert(ori,item)插入元素并能保证元素的升序排列
                bisect.insort(self.centroidDistance[point.id], (dist, pos))
            # 	如果》=，进行比较，如果记住的最大的距离>目前dist,则插入dist以及pos,删掉最后一个
            # 	如何保证插入的len-1就是dist最大的那个？？==好像不能保证，但是能保证最后一个是在慢慢的变小的
            elif self.centroidDistance[point.id][self.centroidsToRemember - 1][0] > dist:
                bisect.insort(self.centroidDistance[point.id], (dist, pos))
                del self.centroidDistance[point.id][self.centroidsToRemember]
            if minDist == -1:
                minDist = dist
                closestCentroid = pos
            elif minDist > dist:
                minDist = dist
                closestCentroid = pos
            pos += 1
        # 	返回的是该点的最小dist以及pos
        return (closestCentroid, minDist)

    # 返回该点的中心，给每一个对应的点都选择一个中心
    def getCentroid(self, point):
        pos = 0
        # dist表示该点与之前中心点的距离
        dist = self.getDistance(point, self.centroidList[point.centroid].point)
        minDist = dist
        # 最近的和当前的中心点，也就是标志它属于那个cluster
        closestCentroid = point.centroid
        currCentroid = point.centroid
        for x in self.initPointList[point.id]:
            centroid = self.centroidList[x]  # 计算该点（是点序列里面的点）与point的距离
            if x != currCentroid:  # 距离最近的和当前的中心不同，更新
                dist = self.getDistance(point, centroid.point)
                if minDist > dist:
                    minDist = dist
                    closestCentroid = x
            pos += 1
        # self.closestClusterDistance[point.id] = minDist
        return (closestCentroid, minDist)

    # 重新选择中心，坐标平均
    def reCalculateCentroid(self):
        pos = 0
        for centroid in self.centroidList:
            zeroArr = []
            for x in range(0, self.dim):
                zeroArr.append(0)
            mean = Point(zeroArr, self.dim)
            for point in centroid.pointList:
                for x in range(0, self.dim):
                    mean.coordinates[x] += point.coordinates[x]
            for x in range(0, self.dim):
                try:
                    mean.coordinates[x] = mean.coordinates[x] / len(centroid.pointList)
                except:
                    mean.coordinates[x] = 0
            centroid.update(mean)
            # centroidList[pos]字典，代表第几簇的位置中心是centroid
            self.centroidList[pos] = centroid
            pos += 1

    # 分配点
    def assignPointsInit(self):
        self.initPointList = {}  # 表示最初的中心节点的点列
        # i表示要遍历全部的点
        for i in range(len(self.pointList) - 1, -1, -1):
            # temp表示距离pointList[i]最近的点的坐标和dist
            temp = self.getCentroidInit(self.pointList[i])
            self.initPointList[self.pointList[i].id] = []
            # 这用到了centroidsToRemember记住的n个节点
            for l in range(0, self.centroidsToRemember):
                # 加上距离该点最近的l个的点==initPointList里面的东西
                self.initPointList[self.pointList[i].id].append(self.centroidDistance[self.pointList[i].id][l][1])
            centroidPos = temp[0]
            centroidDist = temp[1]
            # self.closestClusterDistance[self.pointList[i].id] = centroidDist
            # 第一次，某点的中心节点为空，1、分配中心节点 2、中心节点的点列加点
            if self.pointList[i].centroid is None:
                self.pointList[i].centroid = centroidPos
                self.centroidList[centroidPos].pointList.append(copy.deepcopy(self.pointList[i]))

    def assignPoints(self):
        doneMap = {}
        for i in range(len(self.centroidList) - 1, -1, -1):
            for j in range(len(self.centroidList[i].pointList) - 1, -1, -1):
                try:
                    a = doneMap[self.centroidList[i].pointList[j].id]
                except:
                    doneMap[self.centroidList[i].pointList[j].id] = 1
                    temp = self.getCentroid(self.centroidList[i].pointList[j])
                    centroidPos = temp[0]
                    centroidDist = temp[1]
                    # 如果当前点的中心节点不是距离最近的，1、节点的中心节点换  2、中心节点点的序列增，删
                    if self.centroidList[i].pointList[j].centroid != centroidPos:
                        self.centroidList[i].pointList[j].centroid = centroidPos
                        self.centroidList[centroidPos].pointList.append(
                            copy.deepcopy(self.centroidList[i].pointList[j]))
                        del self.centroidList[i].pointList[j]

    def calculateError(self, config):
        error = 0
        for centroid in self.centroidList:
            for point in centroid.pointList:
                error += self.getDistance(point, centroid.point) ** 2
        return error

    def errorCount(self):
        self.t = threading.Timer(0.5, self.errorCount)
        self.t.start()
        startTime = time.time()
        timeStamp = 0
        if self.error != None:
            timeStamp = math.log(self.error)
        endTime = time.time()
        self.errorList.append(timeStamp)
        self.ti += 0.5

    def mainFunction(self):
        self.iteration = 1
        self.ti = 0.0
        self.errorCount()
        error1 = 2 * self.kmeansThreshold + 1
        error2 = 0
        iterationNo = 0
        self.currentTime = time.time()
        self.startTime = time.time()
        self.assignPointsInit()
        print("First Step:", time.time() - self.startTime)
        while (abs(error1 - error2)) > self.kmeansThreshold:
            iterationNo += 1
            self.iteration = iterationNo
            error1 = self.calculateError(self.centroidList)
            self.error = error1
            print("Iteration:", iterationNo, "Error:", error1)
            self.reCalculateCentroid()
            self.assignPoints()
            error2 = self.calculateError(self.centroidList)
            self.error = error2

        self.t.cancel()


def makeRandomPoint(n, lower, upper):
    return numpy.random.normal(loc=upper, size=[lower, n])


if __name__ == '__main__':
    pointList = []
    x = []
    y = []
    c = []
    numPoints = 10000
    dim = 10
    numClusters = 100
    k = 0
    for i in range(0, numClusters):
        num = int(numPoints / numClusters)
        p = makeRandomPoint(dim, num, k)
        k += 5
        pointList += p.tolist()

    start = time.time()
    # self, k, pointList, kmeansThreshold, predictionThreshold, isPrediction = 0, initialCentroids = None
    # 所以该记住多少个点还有待考量，
    config = Kmeans(numClusters, pointList, 1000, 5)
    print("Time taken:", time.time() - start)
