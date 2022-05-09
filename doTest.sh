#!/bin/sh

echo "Can't Scale Cluster"
echo 

echo "python CYBR_Tlsh_cant_scale.py -f dataDir/mb_25000.csv"
      python CYBR_Tlsh_cant_scale.py -f dataDir/mb_25000.csv
echo

echo "python CYBR_Ssdeep_cant_scale.py -f dataDir/mb_25000.csv"
      python CYBR_Ssdeep_cant_scale.py -f dataDir/mb_25000.csv
echo
echo

echo "Can't Cluster"
echo 

echo "python CYBR_Tlsh_cant_cluster.py -f dataDir/mb_100000.csv"
      python CYBR_Tlsh_cant_cluster.py -f dataDir/mb_100000.csv
echo

echo "python CYBR_Ssdeep_cant_cluster.py -f dataDir/mb_100000.csv"
      python CYBR_Ssdeep_cant_cluster.py -f dataDir/mb_100000.csv
echo

echo "Good Cluster"
echo 

echo "python CYBR_Tlsh_scale.py -f dataDir/mb_100000.csv"
      python CYBR_Tlsh_scale.py -f dataDir/mb_100000.csv
echo

echo "python CYBR_Ssdeep_scale.py -f dataDir/mb_100000.csv"
      python CYBR_Ssdeep_scale.py -f dataDir/mb_100000.csv
echo