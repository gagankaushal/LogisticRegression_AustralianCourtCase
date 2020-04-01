from __future__ import print_function
import sys
import re
from operator import add
import numpy as np 
from pyspark import SparkContext

if __name__ == "__main__":

    sc = SparkContext(appName="LogisticRegression")
    
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

    # Filter out the required five words
    theFiveWords = dictionary.filter(lambda x: x[0] in {'applicant', 'and', 'attack', 'protein', 'car'})
    listofFInalFiveWordsWithCount = theFiveWords.collect()
    
    # Set of required words
    requiredWords  = {'applicant', 'and', 'attack', 'protein', 'car'}

    print ('#'*5,'Word Postions in our Dictionary:','#'*5,'\n')

    # List to store the position of each of the five words
    IntermediateResult = []

    # Function to find and print the relevant positions of words from our dictionary
    def findPos(requiredWords, listofFInalFiveWordsWithCount):
      for i in requiredWords:
        # Initialize a variable to check whether the required word is found or not
        currItemFound = 0

        # Initialize the position as -1
        positionFound = -1

        for j in range(len(listofFInalFiveWordsWithCount)):
          # If the required word is found, change the value of the flag and assign the position found
          if i in listofFInalFiveWordsWithCount[j]:
            positionFound = listofFInalFiveWordsWithCount[j][1]
            currItemFound = 1
            
        IntermediateResult.append((i, '->', positionFound))

        # Print the positions and the required word
        print(i, '->', positionFound)
      return IntermediateResult

    # Call the function to print answers for task 1
    ansForTask1 = findPos(requiredWords, listofFInalFiveWordsWithCount)

    # Save the results in a file
    sc.parallelize(ansForTask1).coalesce(1, shuffle = False).saveAsTextFile(sys.argv[2])

    sc.stop()
