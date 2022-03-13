#!/bin/sh

echo "python CYBR7901_Evaluation_Result_Tlsh.py -f dataDir/mb_10K.csv"
      python CYBR7901_Evaluation_Result_Tlsh.py -f dataDir/mb_10K.csv
echo

echo "python CYBR7901_Evaluation_Result_Ssdeep.py -f dataDir/mb_10K.csv"
      python CYBR7901_Evaluation_Result_Ssdeep.py -f dataDir/mb_10K.csv
echo

