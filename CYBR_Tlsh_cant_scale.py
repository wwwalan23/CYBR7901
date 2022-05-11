import argparse
import sys
import time

import pandas as pd
from sklearn import metrics
import numpy as np

from pylib.tlsh_lib import *
import warnings
warnings.filterwarnings("ignore")

###################################################
# List of Function
def getResult(hashType, clusterType, labelList, clusterNumber):
    data = tlist2cdata(hashList)
    
    d = {word: key for key, word in enumerate(set(labelList))}
    labelList_id = [d[word] for word in labelList]
    #print("labelList_id=", labelList_id)
    
    outlierRemoveLabel = []
    outlierRemoveID = []
    outlierRemoveData = []
    
    for i in range(len(clusterNumber)):
        if clusterNumber[i] >= 0:
            outlierRemoveLabel.append(clusterNumber[i])
            outlierRemoveID.append(labelList_id[i])
            outlierRemoveData.append(data[i])
            
    #print("cluster labels=",clusterNumber)
    #print("outlierRemoveLabel =", outlierRemoveLabel)
    #print("outlierRemoveID =", outlierRemoveID)
    
    #print("Number of cluster labels=", len(clusterNumber))
    #print("Number of outlierRemoveLabel =", len(outlierRemoveLabel))
    
    # Number of decimal place for score
    dp = 4 
    
    homo = round(metrics.homogeneity_score(outlierRemoveID, outlierRemoveLabel), dp)
    silh1 = round(metrics.silhouette_score(data, clusterNumber, metric=sim), dp)
    silh2 = round(metrics.silhouette_score(outlierRemoveData, outlierRemoveLabel, metric=sim), dp)
    cali = round(metrics.calinski_harabasz_score(outlierRemoveData, outlierRemoveLabel), dp)
    dav = round(metrics.davies_bouldin_score(outlierRemoveData, outlierRemoveLabel), dp)
    
    print(clusterType + " ran in " + str(end) + " seconds")
    print("Homogeneity score =",homo)
    print("Silhouette score =",silh1)
    print("Silhouette score with Outlier Remove =",silh2)
    print("Calinski harabasz score =",cali)
    print("Davies bouldin score =",dav)
    print()
    
    result = {"nSample": int(len(tlist)),
              "Hash": str(hashType),
              "Cluster": str(clusterType),
              "nLabel": int(nlabel),
              "nCluster": int(max(clusterNumber)),
              "Time(s)": float(end),
              "Homo.": float(homo),
              "Sil.": float(silh2),
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
df = pd.DataFrame()

(path, file) = datafile.split("/")  # save file path
(filename, filetype) = file.split(".")  # save file type

(tlist, [labelList, dateList, slist]) = tlsh_csvfile(datafile)  # return (tlist, [labelList, dateList, hashList])

hashList = tlist
print("Number of samples is " + str(len(hashList)))
print("Number of Unique Label is " + str(len(set(labelList))))
print("Example hash: " + str(hashList[0]))
nlabel = len(set(labelList))
nClusters = [nlabel]
      
###################################################
# Affinity Propagation
try:
    start = time.perf_counter()
    res = runAffinityPropagation(hashList, random_state=5)
    end = round(time.perf_counter() - start, 4)

    dict = getResult("tlsh", "ap", labelList, res.labels_)
    df = pd.concat((df, pd.DataFrame([dict])), ignore_index=True)

except Exception as e:
    print("Affinity Propagation didn't work.")
    print(e)

###################################################
# Agglomerative Clustering
try:
    start = time.perf_counter()
    res = assignCluster(hashList, nlabel)
    end = round(time.perf_counter() - start, 4)

    dict = getResult("tlsh", "ac", labelList, res.labels_)
    df = pd.concat((df, pd.DataFrame([dict])), ignore_index=True)

except Exception as e:
    print("Agglomerative Clustering didn't work.")
    print(e)

###################################################
# Spectral Clustering
try:
    start = time.perf_counter()
    res = runSpectral(hashList, n_clusters=nlabel)
    end = round(time.perf_counter() - start, 4)

    dict = getResult("tlsh", "sp", labelList, res.labels_)
    df = pd.concat((df, pd.DataFrame([dict])), ignore_index=True)

except Exception as e:
    print("Spectral Clustering didn't work.")
    print(e)

###################################################
# Output
outfile = path + "/output/" + filename + "_Tlsh_result_cs.csv"
df.to_csv(outfile, index=False)

toc = round(time.perf_counter() - tic, 4)
print(df)
print("All code ran in " + str(toc) + " seconds")
