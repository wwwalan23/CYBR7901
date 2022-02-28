

# run test on different input ssdeep and tlsh
def runEvaluation():
    sampleSize = ["1K", "10K", "100K"]
    for x in sampleSize:
        filename = "dataDir/" + "mb_" + sampleSize + ".csv"
        exec(open("CYBR7901_Evaluation_Result_Tlsh.py").read())
        exec(open("CYBR7901_Evaluation_Result_Ssdeep.py").read())

if __name__ == "__main__":
    runEvaluation()
    print("Everything passed")


