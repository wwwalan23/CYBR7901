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

    result = {"nSample": int(len(tlist)),
              "Hash": str(hashType),
              "Cluster": str(clusterType),
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
n = [nlabel]

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

except Exception as e:
    print("Affinity Propagation didn't work.")
    print(e)

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

except Exception as e:
    print("Agglomerative Clustering didn't work.")
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

except Exception as e:
    print("Spectral Clustering didn't work.")
    print(e)

###################################################
# Output
outfile = path + "/output/" + filename + "_Ssdeep_result_cs.csv"
df.to_csv(outfile, index=False)

toc = round(time.perf_counter() - tic, 4)
print(df)
print("All code ran in " + str(toc) + " seconds")
