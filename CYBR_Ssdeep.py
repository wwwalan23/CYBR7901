import argparse
import sys
import time

import pandas as pd
from sklearn import metrics
import numpy as np

from pylib.ssdeep_lib import *
import warnings
warnings.filterwarnings("ignore")

###################################################
# List of Function
def containNan(labelList):
    for label in labelList:
        if label == 'n/a':
            haveNan = True
            return haveNan


def removeNan(hashlist, labelList):
    count = -1
    newhashlist = []
    newlabelList = []

    for label in labelList:
        count += 1
        if label != 'n/a':
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

    # print("Homogeneity score is " + str(homo))
    # print("Silhouette score is " + str(silh))
    # print("Calinski harabasz score is " + str(cali))
    # print("Davies bouldin score is " + str(dav))

    result = {"File": str(filename),
              "nSample": int(len(tlist)),
              "Hash": str(hashType),
              "Cluster": str(clusterType),
              "Has_n/a": bool(haveNan),
              "nLabel": int(nlabel),
              "nCluster": int(max(clusterNumber)),
              "Time(s)": float(end),
              "Homo.": float(homo),
              "Sil.": float(silh),
              "Cal.": float(cali),
              "Dav.": float(dav)}
    return result


###################################################
# start of main program
###################################################

parser = argparse.ArgumentParser(prog='readcsv')  # provides a convenient interface to handle command-line arguments.
parser.add_argument('-f', help='fname', type=str, default="")  # the extra part need to run file
args = parser.parse_args()
datafile = args.f
### datafile = "dataDir/mb_1K.csv" #<-----Change this file size
if (datafile == ""):
    print("you must provide a datafile name (-f)\n")
    sys.exit()
# end if

###################################################

tic = time.perf_counter()  # experiment time counter
haveNan = False
df = pd.DataFrame()

(path, file) = datafile.split("/")  # save file path
(filename, filetype) = file.split(".")  # save file type

(tlist, [labelList, dateList, slist]) = tlsh_csvfile(datafile)  # return (tlist, [labelList, dateList, hashList])
# (tlist, labelList) = tlsh_csvfile(datafile)

# remove Nan Value
# (tlist, labelList) = removeNan(tlist, labelList)

hashList = slist
print("Number of samples is " + str(len(hashList)))
print("Number of Unique Label is " + str(len(set(labelList))))
print("Example hash: " + str(hashList[0]))
nlabel = len(set(labelList))
haveNan = containNan(labelList)

###################################################
# Agglomerative Clustering
try:
    start = time.perf_counter()
    res = assignCluster(hashList, nlabel)
    end = round(time.perf_counter() - start, 4)

    dict = getResult("ssdeep", "ac", labelList, res.labels_)
    df = pd.concat((df, pd.DataFrame([dict])), ignore_index=True)
    print(dict.get('Cluster'))
    print("Code ran in " + str(end) + " seconds")

    #outfile = path + "/output/" + filename + "_hac_out.txt"
    #outputClusters(outfile, hashList, res.labels_, labelList, quiet=True)
except Exception as e:
    print("Agglomerative Clustering didn't work.")
    print(e)

###################################################
# HAC-T
#from pylib.hac_lib import *
"""
try:
    hac_resetDistCalc()

    start = time.perf_counter()
    res = HAC_T(datafile, CDist=30, step3=0, outfname="tmp.txt", cenfname="tmp2.txt")
    end = round(time.perf_counter() - start, 4)

    dict = getResult("ssdeep", "hac-t", labelList, res)
    df = pd.concat((df, pd.DataFrame([dict])), ignore_index=True)
    print(dict.get('Cluster'))
    print("Code ran in " + str(end) + " seconds")

    nclusters = max(res)
    nDistCalc = hac_lookupDistCalc()
    print("nclusters is " + str(nclusters))
    print("nDistCalc is " + str(nDistCalc))

    # outfile = path + "/output/" + filename + "_hac-t_out.txt"
    # outputClusters(outfile, hashList, res, labelList, quiet=True)
except Exception as e:
    print("HAC-T didn't work.")
    print(e)
"""
###################################################
# DBSCAN
try:
    resetDistCalc()

    start = time.perf_counter()
    res = runDBSCAN(hashList, eps=30, min_samples=2, algorithm='auto')
    end = round(time.perf_counter() - start, 4)

    dict = getResult("ssdeep", "dbscan", labelList, res.labels_)
    df = pd.concat((df, pd.DataFrame([dict])), ignore_index=True)
    print(dict.get('Cluster'))
    print("Code ran in " + str(end) + " seconds")

    nclusters = max(res.labels_)
    nDistCalc = lookupDistCalc()
    print("nclusters is " + str(nclusters))
    print("nDistCalc is " + str(nDistCalc))

    # outfile = path + "/output/" + filename + "_dbscan_out.txt"
    # outputClusters(outfile, hashList, res.labels_, labelList, quiet=True)
except Exception as e:
    print("DBSCAN didn't work.")
    print(e)

###################################################
# KMeans
try:
    start = time.perf_counter()
    res = runKMean(hashList, n_clusters=nlabel)
    end = round(time.perf_counter() - start, 4)

    dict = getResult("ssdeep", "kmeans", labelList, res.labels_)
    df = pd.concat((df, pd.DataFrame([dict])), ignore_index=True)
    print(dict.get('Cluster'))
    print("Code ran in " + str(end) + " seconds")

    # outfile = path + "/output/" + filename + "_kmean_out.txt"
    # outputClusters(outfile, hashList, res.labels_, labelList)
except Exception as e:
    print("KMean didn't work.")
    print(e)

###################################################
# Affinity Propagation
try:
    start = time.perf_counter()
    res = runAffinityPropagation(hashList, random_state=5)
    end = round(time.perf_counter() - start, 4)

    dict = getResult("ssdeep", "ap", labelList, res.labels_)
    df = pd.concat((df, pd.DataFrame([dict])), ignore_index=True)
    print(dict.get('Cluster'))
    print("Code ran in " + str(end) + " seconds")

    # outfile = path + "/output/" + filename + "_ap_out.txt"
    # outputClusters(outfile, hashList, res.labels_, labelList)
except Exception as e:
    print("Affinity Propagation didn't work.")
    print(e)

###################################################
# Mean Shift
try:
    start = time.perf_counter()
    res = runMeanShift(hashList, bandwidth=5)
    end = round(time.perf_counter() - start, 4)

    dict = getResult("ssdeep", "ms", labelList, res.labels_)
    df = pd.concat((df, pd.DataFrame([dict])), ignore_index=True)
    print(dict.get('Cluster'))
    print("Code ran in " + str(end) + " seconds")

    # outfile = path + "/output/" + filename + "_ms_out.txt"
    # outputClusters(outfile, hashList, res.labels_, labelList)
except Exception as e:
    print("Mean Shift didn't work.")
    print(e)

###################################################
# Spectral Clustering
try:
    start = time.perf_counter()
    res = runSpectral(hashList, n_clusters=nlabel)
    end = round(time.perf_counter() - start, 4)

    dict = getResult("ssdeep", "sp", labelList, res.labels_)
    df = pd.concat((df, pd.DataFrame([dict])), ignore_index=True)
    print(dict.get('Cluster'))
    print("Code ran in " + str(end) + " seconds")

    # outfile = path + "/output/" + filename + "_sp_out.txt"
    # outputClusters(outfile, hashList, res.labels_, labelList)
except Exception as e:
    print("Spectral Clustering didn't work.")
    print(e)

###################################################
# OPTICS
try:
    start = time.perf_counter()
    res = runOPTICS(hashList, min_samples=2)
    end = round(time.perf_counter() - start, 4)

    dict = getResult("ssdeep", "optics", labelList, res.labels_)
    df = pd.concat((df, pd.DataFrame([dict])), ignore_index=True)
    print(dict.get('Cluster'))
    print("Code ran in " + str(end) + " seconds")

    # outfile = path + "/output/" + filename + "_optics_out.txt"
    # outputClusters(outfile, hashList, res.labels_, labelList)
except Exception as e:
    print("OPTICS didn't work.")
    print(e)
###################################################
# BIRCH
try:
    start = time.perf_counter()
    res = runBIRCH(hashList, n_clusters=nlabel)
    end = round(time.perf_counter() - start, 4)

    dict = getResult("ssdeep", "birch", labelList, res.labels_)
    df = pd.concat((df, pd.DataFrame([dict])), ignore_index=True)
    print(dict.get('Cluster'))
    print("Code ran in " + str(end) + " seconds")

    # outfile = path + "/output/" + filename + "_birch_out.txt"
    # outputClusters(outfile, hashList, res.labels_, labelList)
except Exception as e:
    print("BIRCH didn't work.")
    print(e)
###################################################
# Output
outfile = path + "/output/" + filename + "_Ssdeep_result.csv"
df.to_csv(outfile, index=False)

toc = round(time.perf_counter() - tic, 4)
print(df)
print("All code ran in " + str(toc) + " seconds")
