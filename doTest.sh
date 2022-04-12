#!/bin/sh

echo "Can't Scale Cluster"
echo 

echo "python CYBR_Tlsh_cant_scale.py -f dataDir/mb_25K.csv"
      python CYBR_Tlsh_cant_scale.py -f dataDir/mb_25K.csv
echo

echo "python CYBR_Ssdeep_cant_scale.py -f dataDir/mb_25K.csv"
      python CYBR_Ssdeep_cant_scale.py -f dataDir/mb_25K.csv
echo
echo

echo "Can't Cluster"
echo 

echo "python CYBR_Tlsh_cant_cluster.py -f dataDir/mb_250K.csv"
      python CYBR_Tlsh_cant_cluster.py -f dataDir/mb_250K.csv
echo

echo "python CYBR_Ssdeep_cant_cluster.py -f dataDir/mb_250K.csv"
      python CYBR_Ssdeep_cant_cluster.py -f dataDir/mb_250K.csv
echo

echo "Good Cluster"
echo 

echo "python CYBR_Tlsh_scale.py -f dataDir/mb_250K.csv"
      python CYBR_Tlsh_scale.py -f dataDir/mb_250K.csv
echo

echo "python CYBR_Ssdeep_scale.py -f dataDir/mb_250K.csv"
      python CYBR_Ssdeep_scale.py -f dataDir/mb_250K.csv
echo