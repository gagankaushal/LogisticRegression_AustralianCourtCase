"""
The following program leverages the regression coefficients generated after training the model in task 2 as an input file
"""
from __future__ import print_function
import sys
import re
from operator import add
import numpy as np 
from pyspark import SparkContext

if __name__ == "__main__":

    sc = SparkContext(appName="LogisticRegression_task3")
    
    # Read the dataset 
    d_corpus = sc.textFile(sys.argv[1])
    
    # Each entry in validLines will be a line from the text file
    validDocLines = d_corpus.filter(lambda x : 'id' in x and 'url=' in x)

    # Now, we transform it into a set of (docID, text) pairs
    keyAndText = validDocLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6])) 

    # leveraged the code from assignment 2
    # remove all non letter characters
    regex = re.compile('[^a-zA-Z]')
    keyAndWordsList = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
    
    # Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
    # to ("word1", 1) ("word2", 1)...
    conslidatedWords = keyAndWordsList.flatMap(lambda x: x[1]).map(lambda x: (x,1))

    # Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
    allCounts = conslidatedWords.reduceByKey(add)

    # Get the top 20,000 words in a local array in a sorted format based on frequency
    topWordsinDict = allCounts.top(20000, key = lambda x : x[1])

    # We'll create a RDD that has a set of (word, dictNum) pairs
    # start by creating an RDD that has the number 0 through 20000
    # 20000 is the number of words that will be in our dictionary
    top20000Words = sc.parallelize(range(20000))

    # Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
    # ("NextMostCommon", 2), ...
    # the number will be the spot in the dictionary used to tell us
    # where the word is located
    dictionary = top20000Words.map (lambda x : (topWordsinDict[x][0], x))
    
    
    # The following function gets a list of dictionaryPos values,
    # and then creates a TF vector
    # corresponding to those values... for example,
    # if we get [3, 4, 1, 1, 2] we would in the
    # end have [0, 2/5, 1/5, 1/5, 1/5] because 0 appears zero times,
    # 1 appears twice, 2 appears once, etc.

    def buildArray(listOfIndices):
        
        returnVal = np.zeros(20000)
        
        for index in listOfIndices:
            returnVal[index] = returnVal[index] + 1
        
        mysum = np.sum(returnVal)
        
        returnVal = np.divide(returnVal, mysum)
        
        return returnVal
        
    def getLabel(x):
      if x[:2] == 'AU':
        return 1

      else:
        return 0
    
    
    # Leverage the regression coefficients genereated by task2 (model training) to make the prediction
    #filePathOutputTask2 = sys.argv[2]
    
    # Open the file containing regression coefficients and read it
    
    #task2Lines = filePathOutputTask2.map(lambda x: x.split("Final Regression Coeffients:\n["))
    filePathOutputTask2 =sc.textFile(sys.argv[2])
    
    #filePathOutputTask2.map(lambda x: )
    #with open(filePath) as file:
     #   allLines = file.read()
     
    # Extract out all of the lines present in the output of task 2
    task2Lines = filePathOutputTask2.map(lambda x: x.split(","))
    
    # Extract the line containing the regression coefficients and remove '[' and ']' from the extremes
    listOfLines = task2Lines.collect()[10]
    listOfLines[0] = listOfLines[0][1:]
    listOfLines[len(listOfLines)-1] = listOfLines[len(listOfLines)-1][:len(listOfLines[len(listOfLines)-1])-2]

    # Convert the list of regression coefficients to numpy array to be used as an input for prediction in task 3
    regressionCoefficients = np.array(listOfLines, dtype = np.float64 )
        
    # Split the file and extract the 'Regression Coefficients'    
    #listOfLines = allLines.split("Final Regression Coeffients:\n[")
    #listOfLines[len(listOfLines)-1] = listOfLines[len(listOfLines)-1][:len(listOfLines[len(listOfLines)-1])-2]
    #regressionCoefficients = np.array(listOfLines[1].split(','), dtype = np.float64 )
    
    # Threshold for logistic regression
    threshold = 0.3

    # Prediction Function using logistic regression
    def predictionLogisticRegresison(x):
      value = 1/(1+np.exp(-(  np.dot( x, regressionCoefficients )  )  )) 
      # return value
      if value >= threshold:
        return 1
      else:
        return 0
        
    ###################################################### PREDICTION/ EVALUATION - TAKS 3 ########
    # Read the dataset 
    testData = sc.textFile(sys.argv[3])

    # Each entry in validLines will be a line from the text file
    validDocLinesTest = testData.filter(lambda x : 'id' in x and 'url=' in x)

    # Now, we transform it into a set of (docID, text) pairs
    keyAndTextTest = validDocLinesTest.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6])) 

    # remove all non letter characters
    keyAndWordsListTest = keyAndTextTest.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

    # Get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
    # ("word1", docID), ("word2", docId), ...
    allWordsWithDocIDTest = keyAndWordsListTest.flatMap(lambda x: ((j, x[0]) for j in x[1]))

    # Join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
    allDictionaryWordsTest = dictionary.join(allWordsWithDocIDTest)

    # Drop the actual word itself to get a set of (docID, dictionaryPos) pairs
    justDocAndPosTest = allDictionaryWordsTest.map(lambda x: (x[1][1],x[1][0]))

    # Get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    allDictionaryWordsInEachDocTest = justDocAndPosTest.groupByKey()

    # The following line this gets us a set of
    # (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    # and converts the dictionary positions to a bag-of-words numpy array...
    allDocsAsNumpyArraysTest = allDictionaryWordsInEachDocTest.map(lambda x: (x[0], buildArray(x[1])))

    # Now, create a version of allDocsAsNumpyArrays where, in the array,
    # every entry is either zero or one.
    # A zero means that the word does not occur,
    # and a one means that it does.
    zeroOrOneTest = allDocsAsNumpyArraysTest.map(lambda x: (x[0],np.where(x[1] > 0, 1, 0)))
    
    # Create a RDD of testing data and derive features and labels ... x[0]-> label, x[1]-> features
    yLabelAndXFeatures = zeroOrOneTest.map(lambda x: (getLabel(x[0]),x[1]))
    
    # Make the prediction using the function 'predictionLogisticRegresison'
    yLabelAndXFeaturesPrediction = yLabelAndXFeatures.map(lambda x: (x[0],x[1],predictionLogisticRegresison(x[1])))

    # Function to calculate True Positives
    def calculateTruePositives(x):
      if (x[0] == 1 and x[2] == 1): # the article was Australian court case (x[0]) and the prediction was also Australian court case x[2]
        return 1
      else:
        return 0

    # Function to calculate False Positives
    def calculateFalsePositives(x):
      if (x[0] == 0 and x[2] == 1): # the article was not Australian court case (x[0]) but the prediction was Australian court case x[2]
        return 1
      else:
        return 0

    # Function to calculate False Negatives
    def calculateFalseNegatives(x):
      if (x[0] == 1 and x[2] == 0): # the article was Australian court case (x[0]) but the prediction was not Australian court case x[2]
        return 1
      else:
        return 0
    
    # Function to calculate True Negatives
    def calculateTrueNegatives(x):
      if (x[0] == 0 and x[2] == 0): # the article was not Australian court case (x[0]) and the prediction was not Australian court case x[2]
        return 1
      else:
        return 0

    # Out of total positive labels predicted, how many correctly classified as positive, that is PPV
    def precision(x):
      # Number of true positives/ (Number of true positives + Number of false positives) 
      # return truePositive/(truePositive + falsePositive)
      return x[1][0]/(float(x[1][0] + x[1][1]))

    # Out of actual positive labels, how many correctly classified as positive, that is, TPR
    def recall(x):
      # Number of true positives/ (Number of true positives + Number of false Negatives) 
      # return truePositive/(truePositive + falseNegative)
      return x[1][0]/(float(x[1][0] +  x[1][2]))
      
      
    # Calculate 'True Positives', 'False Positives' and 'False Negatives'
    calcTP_FP_FN = yLabelAndXFeaturesPrediction.map(lambda x: (1, np.array([calculateTruePositives(x), calculateFalsePositives(x), calculateFalseNegatives(x),calculateTrueNegatives(x)]))).reduceByKey(np.add)
    
    print('')
    print ('#'*20)
    print('Number of True Positives:', calcTP_FP_FN.collect()[0][1][0])
    print('Number of False Positives:', calcTP_FP_FN.collect()[0][1][1])
    print('Number of False Negatives:', calcTP_FP_FN.collect()[0][1][2])
    print('Number of True Negatives:', calcTP_FP_FN.collect()[0][1][3])
    print('')
    
    
    # if 'Number of True Positives: 0 and 'Number of False Positives: 0, then F1 score is N/A
    if calcTP_FP_FN.collect()[0][1][0] == 0  and calcTP_FP_FN.collect()[0][1][1] == 0:
        calculateF1score = 'N/A'
        print('F1 score for classifier =','N/A')
        print ('#'*20)
        print('')
    else:    
        calculateF1score = calcTP_FP_FN.map(lambda x: (precision(x), recall(x))).map(lambda x: 2*x[0]*x[1] / (x[0] + x[1])).collect()[0]
        print('F1 score for classifier =',round(calculateF1score*100,2),'%')
        print ('#'*20)
        print('')
    
    # List to store the results of task 3
    ansForTask3 = []
    
    if calculateF1score != 'N/A':
        ansForTask3.append(('F1 score for classifier =',round(calculateF1score*100,2),'%'))
    else:
        ansForTask3.append(('F1 score for classifier =','N/A'))
    ansForTask3.append('')
    ansForTask3.append(('Number of True Positives', calcTP_FP_FN.collect()[0][1][0]))
    ansForTask3.append(('Number of False Positives', calcTP_FP_FN.collect()[0][1][1]))
    ansForTask3.append(('Number of False Negatives', calcTP_FP_FN.collect()[0][1][2]))
    ansForTask3.append(('Number of True Negatives', calcTP_FP_FN.collect()[0][1][3]))
    
    # Save the results of task3 in a text file
    sc.parallelize(ansForTask3).coalesce(1, shuffle = False).saveAsTextFile(sys.argv[4]) 
    
    sc.stop()