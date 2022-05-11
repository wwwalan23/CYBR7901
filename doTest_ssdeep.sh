#!/bin/sh

echo "Can't Scale Cluster"
echo "python CYBR_Ssdeep_cant_scale.py -f dataDir/mb_100.csv"
      python CYBR_Ssdeep_cant_scale.py -f dataDir/mb_100.csv
echo

echo "Can't Cluster"
echo "python CYBR_Ssdeep_cant_cluster.py -f dataDir/mb_100.csv"
      python CYBR_Ssdeep_cant_cluster.py -f dataDir/mb_100.csv
echo

echo "Good Cluster"
echo "python CYBR_Ssdeep_scale.py -f dataDir/mb_100.csv"
      python CYBR_Ssdeep_scale.py -f dataDir/mb_100.csv
echo