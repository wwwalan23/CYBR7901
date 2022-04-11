#!/bin/sh

echo "python CYBR_Tlsh_cant_scale.py -f dataDir/mb_70K.csv"
      python CYBR_Tlsh_cant_scale.py -f dataDir/mb_70K.csv
echo

echo "python CYBR_Tlsh_cant_scale.py -f dataDir/mb_60K.csv"
      python CYBR_Tlsh_cant_scale.py -f dataDir/mb_60K.csv
echo

echo "python CYBR_Tlsh_cant_scale.py -f dataDir/mb_50K.csv"
      python CYBR_Tlsh_cant_scale.py -f dataDir/mb_50K.csv
echo

echo "python CYBR_Tlsh_cant_scale.py -f dataDir/mb_40K.csv"
      python CYBR_Tlsh_cant_scale.py -f dataDir/mb_40K.csv
echo

echo "python CYBR_Tlsh_good.py -f dataDir/mb_390K.csv"
      python CYBR_Tlsh_good.py -f dataDir/mb_390K.csv
echo

echo "python CYBR_Ssdeep_goog.py -f dataDir/mb_390K.csv"
      python CYBR_Ssdeep_goog.py -f dataDir/mb_390K.csv
echo