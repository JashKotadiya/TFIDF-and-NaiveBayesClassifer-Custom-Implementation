import pandas
import numpy 
# The pandas python library is used to join merge and concentinate different data sets 
# We need to use a self.dataFrame which is the pandas equivalent of a excel spreadsheet - it has rows and columns 
# The rows have labels for each row whihc are indexes and the columns are named variables 
import re

#################################################
#   Custom Implementation of TFIDF algorithm    # 
##################################################
class TFIDFAnalyzer: 
   def __init__(self) -> None:
      self.cleanedTransactionsList = list()
      self.uniqueWords = set()
      self.wordDictionarySplit = dict() 
      self.termFrequencyMatrix = None
      self.inverseTermFrequencyVector = None 
      self.tfIDF = None 

   def cleanTransaction(self, text: str) -> str: 
      #Convert to lowercase
      text = text.lower()
      # These take the general format of sub(pattern, replacement, text), where pattern is what you wanna find, 
      # replacement is what you replace it with and text is the string you are modifying 
      # For the expression r'[^a-z0-9\s]' this is a regular expression, [ ] means match any charactrs inside ^ means not, 
      # So this pattern is saying match any characters which are not a-z, 0-9 or whitespaces, so special characters get removed
      # Replace with a whitespace instead of empty string so that things like udemy.com.ifzowx and udemy.com.rmff become udemy, com, ifwozw and not udemycomifwox and udemycomrmff because 
      # then those would become seperate words 
      text = re.sub(r'[^a-z0-9\s]', ' ', text)
      # /d means a digit ie 0-9, + means one or more of, so \d+ means one or more digits ie one or more 0-9s, so 
      # this removes any numbers 
      # Replace words with " " and not empty string for the same reason as removing the special characters 
      text = re.sub(r'\d+',' ', text)
      return text
   
   def buildVocabulary(self):
      # Iterates through the cleaned transactions list, instead of a data frame 
      # This was initially implemented as adding the words to a list using the extend() function and then converting to a set() 
      # but this previous method had more space complexitiy because say if you had a column of 100000 words, python would have to copy all of the words you wanted to add using extend() 
      # then when we needed more space python would have to keep asking a for a lot of ram from the computer 
      # All just to delete a lot of the words later 
      for string in self.cleanedTransactionsList:
         words = string.split()
         for word in words: 
            self.uniqueWords.add(word) # type: ignore (the warning is wrong because uniqueWords is initalized to a set and then converted to a list)

      # Sort the set into alphabetical order 
      self.uniqueWords = sorted(list(self.uniqueWords))

      # Convert the set into a dictionary, with key being the word and value the index of the word 
      for i in range(0,len(self.uniqueWords)): 
         self.wordDictionarySplit[self.uniqueWords[i]] = i; 

      # This generates our termFrequency, intially as a matrix of all zeros, because if we instead initalize a numpy arr because 
      # they are fixed size, everytime we have to add a new item we have to copy the entire array which is not good 
      self.termFrequencyMatrix = numpy.zeros((len(self.cleanedTransactionsList),len(self.wordDictionarySplit)))

   # Function to find the number of times term t appears in transaction d 
   # Index is the index of the transaction
   # Deprecated 
   @DeprecationWarning
   def countTermInTransaction(self, t: str, index: int) -> int: 
      return (self.cleanedTransactionsList[index]).count(t)

   # Return number of words in trasnaction t 
   # Index is the index of the transaction
   # Deprecated 
   @DeprecationWarning
   def totalWordsInTransaction(self, index: int) -> int:
      return len((self.cleanedTransactionsList[index]).split())

   # Calculate term frequency - which is the probabilty of a term appearing in a specific document d 
   # The reason we use propportion and not raw count is because raw count for things like uber trip and uber trip san fran airport rode 
   # Raw count would tell us that uber has a count of 1, when uber is clearly more important in uber trip than in the second one 
   # So we use a proportion where the denomenator is the total number of words, and in the first one we get uber is 0.5 where in the second one we get uber is 0.17
   # We wanna know how much of this transaction is in this word 
   # Get each transaction the cleaned column in the data frame, split it into its indivual words
   # Then calculate the length of the transaction 
   # Go through each word in the transaction and see if it is in our mastery dictionary 
   # and if it is increase the count for the specfic row and column 
   # After each transaction, divide by length of the transaction 
   # Repeat for all transactions 
   def termFrequency(self): 
      cleanedTransactionsList = self.cleanedTransactionsList
      for i in range(0, len(cleanedTransactionsList)):
         words = cleanedTransactionsList[i].split()
         totalWordLength = len(words)

         # If there are no words inside this index of the cleanedTransactions list, skip it, otherwise, totalWordLength = 0 and we will be doing division by 0 
         # Instead of crashing, numpy here will fill your matrix with a special value called NaN, and any future math you will do with the matrix will also output NaN
         # Instead we should skip this specefic index, because since the matrix was initalized to all zeros, this index will be 0 correctly
         if totalWordLength == 0: 
            continue

         for word in words: 
            if word in self.wordDictionarySplit: 
               matrixColumnIndex = self.wordDictionarySplit[word]
               self.termFrequencyMatrix[i, matrixColumnIndex] += 1 # type: ignore

         self.termFrequencyMatrix[i] = self.termFrequencyMatrix[i] / totalWordLength # type: ignore


   
   # Create a vector for the inverseTermFrequency
   
   # Need N and the docuement frequency
   # Document frequency is the total number of individual transactions which have the word t 
   # First determine the total length the transactions and then convert to a list 
   # Then loop through each transaction, for each transaction split it into its individual words 
   # Convert to a set so that duplicates words within a transaction are only counted once 
   # N / df determines how rare a word is, N is the total # of transactions and df is the number of transacitons containing the word 
   # If N = 1000 transactions, and if we choose the word payment, say df = 900, then we get 1000 / 900  = 1.11, which is very close to 1, meaning a very common word 
   # The log which is log base e, is used because without the log, rare words would get very big values, so taking the log compresses the scale
   def inverseDocumentFrequency(self): 
      self.inverseTermFrequencyVector = numpy.zeros(len(self.wordDictionarySplit))
      N =  len(self.cleanedTransactionsList)

      cleanedTransactionsList = self.cleanedTransactionsList
      for i in range(0, len(cleanedTransactionsList)):
         words = cleanedTransactionsList[i].split()
         words = set(words)
         if len(words) == 0: 
            continue

         for word in words: 
            if (word in self.wordDictionarySplit):
               self.inverseTermFrequencyVector[self.wordDictionarySplit[word]] += 1 

      return numpy.log(N / self.inverseTermFrequencyVector)
   
   # Here we training the TFIDF anaylzer, we give it the data and it cleans the data 
   # Builds the vocab and calcualtes termFrequency, inverseDocumentFrequency and gives us the final tfIDF matrix
   # This is preparing the training data for the model, the model will have a one to one correspondence with the matrix here, ie its row index and column index 0 will be for the same 
   # word and same transaction 
   def fitTransform(self, X_train_raw: list): 
      self.cleanedTransactionsList = []
      for i in range(0, len(X_train_raw)):
         self.cleanedTransactionsList.append(self.cleanTransaction(X_train_raw[i]))

      self.buildVocabulary()
      self.termFrequency()
   # Final matrix giving us the values for the tfIDF algorithm 
   # We mutiply the tf and idf scores together becuase we want a word to be important iff its rare in in document and its rare in the whole dictionary 
   # tf tells us if a term is rare in a document and idf tells us if the term is rare among all words 
   # Since mutiplication enforces the rule that small x small is small and big times big is big and big x small is small, it keeps rare words rare and common words common
   # We could use addition insetad but mutiplication maeks the difference more clear
      self.inverseTermFrequencyVector = self.inverseDocumentFrequency()
      self.tfIDF = numpy.multiply(self.termFrequencyMatrix, self.inverseTermFrequencyVector) # type: ignore (this warning is wrong because we intialize this matrix always to a numpy array)
      return self.tfIDF
   
   # Here we prepare the testing data to input into the model, 
   def transform(self, X_test_raw: list): 
      self.cleanedTransactionsList = [] # Reset it to empty
      for i in range(0, len(X_test_raw)):
         self.cleanedTransactionsList.append(self.cleanTransaction(X_test_raw[i]))
      self.termFrequencyMatrix = numpy.zeros((len(self.cleanedTransactionsList),len(self.wordDictionarySplit)))
      self.termFrequency()
      # Need to change the second part of this 
      return numpy.multiply(self.termFrequencyMatrix, self.inverseTermFrequencyVector)


   # Example transactions (cleaned):
   # 1: "uber trip"
   # 2: "uber eats"
   # 3: "mcdonalds order"
   # 4: "amazon purchase"
   # 5: "uber ride"

   # Vocabulary (columns of the matrix):
   # [uber, trip, eats, mcdonalds, order, amazon, purchase, ride]

   # Term Frequency (TF) matrix
   # rows = transactions
   # columns = words

   #            uber  trip  eats  mcdonalds  order  amazon  purchase  ride
   # uber trip   0.5  0.5   0     0          0      0       0         0
   # uber eats   0.5  0     0.5   0          0      0       0         0
   # mcd order   0    0     0     0.5        0.5    0       0         0
   # amazon pur  0    0     0     0          0      0.5     0.5       0
   # uber ride   0.5  0     0     0          0      0       0         0.5

   # Example IDF values (rarer words are larger)
   # uber = 0.18
   # trip = 1.10
   # eats = 1.10
   # mcdonalds = 1.10
   # order = 1.10
   # amazon = 1.10
   # purchase = 1.10
   # ride = 1.10

   # TF-IDF matrix (TF * IDF)

   #            uber  trip  eats  mcdonalds  order  amazon  purchase  ride
   # uber trip   0.09 0.55  0     0          0      0       0         0
   # uber eats   0.09 0     0.55  0          0      0       0         0
   # mcd order   0    0     0     0.55       0.55   0       0         0
   # amazon pur  0    0     0     0          0      0.55    0.55      0
   # uber ride   0.09 0     0     0          0      0       0         0.55
   def printResults(self):
      print(pandas.DataFrame(self.tfIDF))
      print(numpy.sum(self.tfIDF[:, 9])) # type: ignore (this warning is wrong)
   
 


# analyzer = TFIDFAnalyzer("bank_transactions.csv")  
# analyzer.fit()
# analyzer.printResults()      



    

      
   

    









