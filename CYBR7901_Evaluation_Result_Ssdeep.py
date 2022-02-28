from sklearn import metrics
from pylib.ssdeep_lib import *
import numpy as np
import pandas as pd
import time

###################################################
# List of Function
def containNan(labelList):
    for lable in labelList:
        if lable == 'n/a':
            haveNan = True
            return haveNan

def removeNan(hashlist, labelList):
    count = -1
    newhashlist = []
    newlabelList = []

    for lable in labelList:
        count += 1
        if lable != 'n/a':
            newhashlist.append(hashlist[count])
            newlabelList.append(labelList[count])
    return newhashlist, newlabelList

def getResult(hashType, clusterType, labelList, clusterNumber):
    d = {word: key for key, word in enumerate(set(labelList))}
    labelList_id = [d[word] for word in labelList]
    a = np.array(labelList_id).reshape(-1, 1)

    # decimal place
    dp = 4

    homo = round(metrics.homogeneity_score(labelList_id, clusterNumber), dp)
    silh = round(metrics.silhouette_score(a, clusterNumber, metric='euclidean'), dp)
    cali = round(metrics.calinski_harabasz_score(a, clusterNumber), dp)
    dav = round(metrics.davies_bouldin_score(a, clusterNumber), dp)

    print("Homogeneity score is " + str(homo))
    print("Silhouette score is " + str(silh))
    print("Calinski harabasz score is " + str(cali))
    print("Davies bouldin score is " + str(dav))

    result = {"File": str(filename),
              "nSample": int(len(tlist)),
              "Hash": str(hashType),
              "Cluster": str(clusterType),
              "Has_n/a": bool(haveNan),
              "nLabel": int(nlable),
              "nCluster": int(max(clusterNumber)),
              "Time(s)": float(end),
              "Homo.": float(homo),
              "Sil.": float(silh),
              "Cal.": float(cali),
              "Dav.": float(dav)}
    return result

###################################################

tic = time.perf_counter() # experiment time counter
haveNan = False
df = pd.DataFrame()

datafile = "dataDir/mb_100.csv" #<-----Change this file size

(path,file) = datafile.split("/") #save file path
(filename,filetype) = file.split(".") #save file type

(tlist, [labelList, dateList, slist]) = tlsh_csvfile(datafile) # return (tlist, [labelList, dateList, hashList])
#(tlist, labelList) = tlsh_csvfile(datafile)

#remove Nan Value
#(tlist, labelList) = removeNan(tlist, labelList)

hashList = slist
print("Number of samples is " + str(len(hashList)))
print("Number of Unique Lable is " + str(len(set(labelList))))
print(hashList[0:10])
nlable = len(set(labelList))
haveNan = containNan(labelList)


###################################################
#Agglomerative Clustering
start = time.perf_counter()
res = assignCluster(hashList, nlable)
end = round(time.perf_counter() - start,4)
print("Code ran in " + str(end) + " seconds")

dict = getResult("ssdeep","hac",labelList,res.labels_)
df = df.append(dict, ignore_index = True)
print(df)

#outfile = path + "/output/" + filename + "_hac_out.txt"
#outputClusters(outfile, hashList, clusterNumber, labelList, quiet=True)

"""
###################################################
# HAC-T
from pylib.hac_lib  import *

hac_resetDistCalc()

start = time.perf_counter()
res = HAC_T(datafile, CDist=30, step3=0, outfname="tmp.txt", cenfname="tmp2.txt")
end = round(time.perf_counter() - start,4)
print("Code ran in " + str(end) + " seconds")

nclusters = max(res)
nDistCalc = hac_lookupDistCalc()
print("Number of cluster is " + str(nclusters))
print("Number of Distance Calculated is " + str(nDistCalc))

dict = getResult("ssdeep","hac-t",labelList,res)
df = df.append(dict, ignore_index = True)
print(df)

#outfile = path + "/output/" + filename + "_hac-t_out.txt"
#outputClusters(outfile, hashList, res, labelList, quiet=True)
"""
###################################################
# DBSCAN

resetDistCalc()

start = time.perf_counter()
res = runDBSCAN(hashList, eps=30, min_samples=2, algorithm='auto')
end = round(time.perf_counter() - start,4)
print("Code ran in " + str(end) + " seconds")

nclusters = max(res.labels_)
nDistCalc = lookupDistCalc()
print("nclusters is " + str(nclusters))
print("nDistCalc is " + str(nDistCalc))

dict = getResult("ssdeep","dbscan",labelList,res.labels_)
df = df.append(dict, ignore_index = True)
print(df)

#outfile = path + "/output/" + filename + "_dbscan_out.txt"
#outputClusters(outfile, hashList, res.labels_, labelList, quiet=True)

###################################################
# KMeans

start = time.perf_counter()
res = runKMean(hashList, nlable)
end = round(time.perf_counter() - start,4)
print("Code ran in " + str(end) + " seconds")

dict = getResult("ssdeep","kmeans",labelList,res.labels_)
df = df.append(dict, ignore_index = True)
print(df)

#outfile = path + "/output/" + filename + "_kmean_out.txt"
#outputClusters(outfile, hashList, res.labels_, labelList)

###################################################
# Affinity Propagation
"""
start = time.perf_counter()
res = runAffinityPropagation(hashList,5)
end = round(time.perf_counter() - start,4)
print("Code ran in " + str(end) + " seconds")

dict = getResult("ssdeep","ap",labelList,res.labels_)
df = df.append(dict, ignore_index = True)
print(df)
"""
###################################################
# Mean Shift

start = time.perf_counter()
res = runMeanShift(hashList,5)
end = round(time.perf_counter() - start,4)
print("Code ran in " + str(end) + " seconds")

dict = getResult("ssdeep","ms",labelList,res.labels_)
df = df.append(dict, ignore_index = True)
print(df)

###################################################
# Spectral Clustering

start = time.perf_counter()
res = runMeanShift(hashList,5)
end = round(time.perf_counter() - start,4)
print("Code ran in " + str(end) + " seconds")

dict = getResult("ssdeep","sp",labelList,res.labels_)
df = df.append(dict, ignore_index = True)
print(df)

###################################################
# OPTICS

start = time.perf_counter()
res = runOPTICS(hashList,2)
end = round(time.perf_counter() - start,4)
print("Code ran in " + str(end) + " seconds")

dict = getResult("ssdeep","OPTICS",labelList,res.labels_)
df = df.append(dict, ignore_index = True)
print(df)

###################################################
# BIRCH

start = time.perf_counter()
res = runBIRCH(hashList,2)
end = round(time.perf_counter() - start,4)
print("Code in " + str(end) + " seconds")

dict = getResult("ssdeep","BIRCH",labelList,res.labels_)
df = df.append(dict, ignore_index = True)
print(df)

###################################################
# Output
#outfile = path + "/output/" + filename + "_result.csv"
#df.to_csv(outfile, index = False)

toc = round(time.perf_counter() - tic,4)
print("All code ran in " + str(toc) + " seconds")
