from __future__ import print_function
import sys
import re
from operator import add
import numpy as np 
from pyspark import SparkContext

if __name__ == "__main__":

    sc = SparkContext(appName="LogisticRegression_task2")
    
    # Read the training dataset 
    d_corpus = sc.textFile(sys.argv[1])
    
    # Each entry in validDocLines will be a line from the text file
    validDocLines = d_corpus.filter(lambda x : 'id' in x and 'url=' in x)

    # Transform it into a set of (docID, text) pairs
    keyAndText = validDocLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6])) 

    # leveraged the code from assignment 2
    # remove all non letter characters
    regex = re.compile('[^a-zA-Z]')
    keyAndWordsList = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
    
    # Get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
    # to ("word1", 1) ("word2", 1)...
    conslidatedWords = keyAndWordsList.flatMap(lambda x: x[1]).map(lambda x: (x,1))

    # Count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
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
        
    # Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
    # ("word1", docID), ("word2", docId), ...
    allWordsWithDocID = keyAndWordsList.flatMap(lambda x: ((j, x[0]) for j in x[1]))

    # Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
    allDictionaryWords = dictionary.join(allWordsWithDocID)
    
    # Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
    justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1],x[1][0]))
    
    # Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
    
    # The following line this gets us a set of
    # (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    # and converts the dictionary positions to a bag-of-words numpy array...
    allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))
    
    # Now, create a version of allDocsAsNumpyArrays where, in the array,
    # every entry is either zero or one.
    # A zero means that the word does not occur,
    # and a one means that it does.
    zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0],np.where(x[1] > 0, 1, 0)))
    
    # Function to generate labels for each document - Document with AU id --> 1, else --> 0
    def getLabel(x):
      if x[:2] == 'AU':
        return 1

      else:
        return 0
    
    # Generate features and labels using the 'get Label' funciton -- x[0] -> Label (0 or 1)      x[1] -> Features for each document
    yLabelAndXFeatures = zeroOrOne.map(lambda x: (getLabel(x[0]),x[1]))
    
    # Cache the RDD that will be used in different iterations (while loop)
    yLabelAndXFeatures.cache()
    
    # Initialize the different variables
    numberOfFeatures = 20000
    learningRate = 0.0000001
    regressionCoefficients = np.zeros(numberOfFeatures)
    gradients = np.zeros(numberOfFeatures)
    totalNumberOfIterations = 400
    currentIteration = 0
    listOfCostOrLoss = []
    oldCost = float("inf")
    lambdaRegularisationCoefficient = 10
    oldregressionCoefficients = np.zeros(numberOfFeatures)
    
    # while loop to train the model
    while (currentIteration < totalNumberOfIterations):

      # Calculation of Cost or loss (Negative LLH)
      lossNegativeLLH = yLabelAndXFeatures.map(lambda x: (1,-x[0]*(np.dot(x[1],regressionCoefficients)) + np.log(1 + np.exp(np.dot(x[1],regressionCoefficients)) ))).reduceByKey(add).collect()[0][1]
      # Calculation of gradients
      gradients = yLabelAndXFeatures.map(lambda x: (1, -x[1]*x[0]  + x[1]*(np.exp(np.dot(x[1],regressionCoefficients)) / (1+np.exp(np.dot(x[1],regressionCoefficients)))) )).reduceByKey(np.add).collect()[0][1]

      # Regularization with lambda = 10
      lossNegativeLLH += lambdaRegularisationCoefficient*np.dot(regressionCoefficients,regressionCoefficients)
      gradients += 2*lambdaRegularisationCoefficient*regressionCoefficients

      # Updating the regression coefficients
      regressionCoefficients -= learningRate*gradients

      print('#'*10, 'Iteration', currentIteration + 1,'#'*10)
      print('Cost:', lossNegativeLLH)
      print('Regression Coeffients:', np.around(regressionCoefficients,6))
      print('L2 norm of difference in parameter vector across iterations:',(np.linalg.norm(np.subtract(oldregressionCoefficients, regressionCoefficients))),'\n')
      
      # BOLD DRIVER
      if (oldCost > lossNegativeLLH):
        learningRate *= 1.05
      else:
        learningRate *= 0.5
      
      # End iteration if l2 norm
      if ((np.linalg.norm(np.subtract(oldregressionCoefficients , regressionCoefficients))) < 0.001):
        print('Training stopped at iteration', currentIteration + 1)
        break
      
      # Store the current regression coefficients as old regression coefficients (for comparing and finding  difference of the l2 norm in the parameter vector across iterations)
      oldregressionCoefficients = np.copy(regressionCoefficients)
      
      # Store the cost in a list (this will be used later on plot a graph = Cost vs Number_of_Iterations)
      listOfCostOrLoss.append(lossNegativeLLH)
      oldCost = lossNegativeLLH
      currentIteration += 1
      
    #np.set_printoptions(threshold=np.inf)
    print('Final Regression Coeffients', np.around(regressionCoefficients,6),'\n')
    
    # Filter out the top 5 words with highest regression coefficients - that are most strongly related with an Australian court case
    top5Words = dictionary.filter(lambda x: x[1] in regressionCoefficients.argsort()[-5:][::-1]).map(lambda x: (x[0], regressionCoefficients[x[1]], x[1])).top(5, key = lambda x:  x[1])

    # Print Top 5 words with largest regression coefficients (five words that are most strongly related with an Australian court case)
    print ('#'*5,'Top 5 Words with the largest regression coefficients:','#'*5)
    for word in top5Words:
        print(word)
    
    # List to store the results of task 2
    ansForTask2 = []
    
    ansForTask2.append(('#'*5,'Top 5 Words with the largest regression coefficients:','#'*5))
    for word in top5Words:
        ansForTask2.append(word)
    ansForTask2.append('')
    ansForTask2.append(('Final Cost:',lossNegativeLLH))
    ansForTask2.append('')
    ansForTask2.append('Final Regression Coeffients:')
    ansForTask2.append(regressionCoefficients.tolist())
    

    # Save the results of task2 in a text file
    sc.parallelize(ansForTask2).coalesce(1, shuffle = False).saveAsTextFile(sys.argv[2]) 
    
    # Save the 'list of cost' for task2 in a text file. This list is used to plot the graph
    sc.parallelize(listOfCostOrLoss).coalesce(1, shuffle = False).saveAsTextFile(sys.argv[3]) 

    sc.stop()