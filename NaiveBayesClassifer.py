import numpy
import pandas
import math
# How this model works: At its core its asking - Given this specific transaction text X what is the probabilty that it belongs to 
# Category C 
# ie P(C | X)
# We can use Baye's thm to calculate this probabilty 
# P(C|X) = ( P(X|C) x P(C) )\ P(X)
# Since for a transaction, we are just trying to find the highest scoring category, P(X) will be the same for all the Categories we test 
# ie P("uber eats") is the same for C = "Food" or C = "Clothing"
# So we can just ignore it and focus on P(X|C) x P(C)
# A transaction is a vector of many words so the problem is calcualting P(X|C), meaning calculating the probabilty of an entire specfic sequence of words toegther is very hard 
# So this algorithm makes a massive and techinally incorrect assumption but it works, it assumes each word is completly independent of the others 
# This means P(X|C) = P(x1|C) x P(x2|C) x P(x3|C) x ... x P(xn|C)
class NaiveBayesClassifer: 
    def __init__(self): 
      self.logPriors = dict() 
      self.logLikelyhoods = dict() 
      self.uniqueCatergories = list()

    # X is the input data here, and y is the correct answer, X is the matrix with all the tf idf scores for training
    # Calculate the probabilty for each word for each category, so you would do Uber and the probabilty for each category 
    # and then move onto the next word 
    # The goal of fit is just to train the model here, we just built the master dictionary for the priors and the likelyhoods and nothing else 
    # No values probabilites are calculated here 
    def fit(self, X: numpy.ndarray, Y: numpy.ndarray): 
        # This gives us the sorted and unique categories of Y 
        self.uniqueCatergories = numpy.unique(Y)
        totalTransactions = len(Y)
        for category in self.uniqueCatergories:
           # Calculate the prior ie P(C) probabilty for the category
           # This is using something called numpy's boolean indexing/boolean masking. Y == category gives us a flattened 
           # 1d array with true or false values depending on the boolean condition Y == category 
           # This condition checks every single value in the matrix regardless of rows or columns 
           # So we can use numpy.sum() on this because it'll only count true values, since in python false is 0 and true is 1 
           prior = numpy.sum(Y == category)/totalTransactions
           self.logPriors[category] = numpy.log(prior)
           # Calculate the likelyhood for the category 
           # Here Y == category gives us a boolean 1d array with true or false values, and when you feed that into a numpy matrix 
           # numpy gets the rows according to that true or false array 
           # So if you had X[[true, false, true]], likelyhood would give you the first row and the third row of X, and it would be kept 
           # as a matrix in the same shape as before also 
           # So this gives us all the transactions which are in the food category 
           # Note that the each transaction maps to exactly one category, and here our category column Y and the transaction columns are made so transaction 0 already is category 0 
           rowsInCategory = X[Y == category]
           # Explaining numerator below: 
           # Summing along axis 0 of the rowsInCategory matrix gives us a 1d vector of the list of the sums of the columns 
           # So we have the numerator of P(xi | C) in each index, so P(x1 | C) in index 0 etc 
           # Adding the one at the end is called laplace smoothing, the reason is because say we wanna find P(Food | uber eats ), this 
           # equals P(uber | Food) x P(eats | Food), now P(uber | Food) = 0 but uber eats should have a good probability of becoming food, so we add one to every item which is in our current category
           numerator = numpy.sum(rowsInCategory, axis=0) + 1 
           # The denomenator is the sum of everythiing in the rowsInCategory matrix which is every number which is the in the current category
           # Adjust the denomenator, because we added one to everything in numerator we have to do the same here which is what X.shape[1] is, it is the number of columns 
           # Note that we divide the entire numerator which is an array of the top part of P(xi | C) by the denomenator 
           # So the denomantor gives us the denomenator for one of the P(xi | C), which is the sum of all of the 
           # values for this category plus the number of columns (the second part is laplace smoothing)
           denomenator = numpy.sum(rowsInCategory) + X.shape[1]
           # Store in the dictionary
           # This stores in the dictionary all of the P(xi | C) for this category 
           # The log of the priors and the log is taken here because suppose we need to calculate P(Uber eats | Food) wiht our model 
           # this is P(Uber | Food) x P(Eats | Food), and if each of these numbers is 0.0001 and 0.00001 respecitly, we would get a really small number 
           # which the computer would just store as zero because it has a finite memory 
           # So instead we do log(P(Uber eats | Food)) = log(P(Uber | Food ) x P(Eats | Food)) = log(P(Uber | Food)) + log(P(Eats |Food))
           # And because probabilties are 1 or less they will be converted by the log to a value of zero or less
           # So by the log product rule which we've done above, we are just adding negative numbers instead of muitplying really small numbers which is better 
           # for the computer and more easy for it to do 
           self.logLikelyhoods[category] = numpy.log(numerator/denomenator)
           # The below is the classic approach for calculating the likelyhoodsbut there is a faster way just using numpy (done above)
         #   for i in range(0, len(self.wordDictionarySplit)):
         #      for j in range(0, len(self.cleaned)):
         #         if (catergoriesList[j] == uniqueCatergoriesList[i]):
         #            sum = sum + numpy.sum(self.tfIDF[j,i])
           
    # Here X should be a tf idf matrix of scores which the model has never seen before 
    # Look at one category for this say C = Food and one transaction 
    # Then to get P(Food | Uber eats) = (P(Food) x P(Uber eats | Food)/P(Food)
    # Since P(Food) doesnt change for this probabilty, and we just wanna see which category is the highest, we can remove it 
    # Because dividing by a different constant for each category (the constant is P(Category)) wont change anything by much 
    # So our formula is now P(Food | Uber eats) = P(Food) x P(Uber eats | Food)
    # The algorithm here makes a wrong assumption that all words are independent from each other, but this works well
    # So our formula is P(Food | Uber eats) = P(Food) x P(Uber | Food) x P(Eats | Food)
    # Now if each of these numbers is 0.0001 and 0.00001 respecitly, we would get a really small number 
    # which the computer would just store as zero because it has a finite memory 
    # So so we take the log of both sides, because we know that log(A x B) = log(A) + log(B), so instead of mutiplying small 
    # number we'll be adding neg numbers which is better
    #  And because probabilties are 1 or less they will be converted by the log to a value of zero or less
    # So our formula is log(P(Food | Uber eats)) = log(P(Food) x P(Uber | Food) x P(Eats | Food))
    # log(P(Food | Uber eats)) = log(P(Food)) + log(P(Uber | Food))  + log(P(Eats | Food))
    # We also have a tf idf matrix, where each word has a weight for its transaction 
    # So we should add that weight to our probabitly in the form of mutiplication 
    # log(P(Food | Uber eats)) = log(P(Food)) + TFIDF(Uber for Uber eats)(log(P(Uber | Food))) + TFIDF(Eats for Uber eats)(log(P(Eats | Food)))
    # Generalizing this for a category Score(C) = log(P(C)) + sum(TFIDF(xi) x log(P(xi | C)) -> We do all the words at once for all transactions 
    # From the math above, we've done the logLikelyhoods dictionary, so that for each category, the words are in the same order as the words in the TF IDF matrix
    def predict(self, X: numpy.ndarray): 
       scores = list()
       for category in self.uniqueCatergories:
          # Get the prior for this category from our training
          prior = self.logPriors[category]
          # Here we take our entire new TF IDF Matrix X and take the dot product with the likelyhoods for the category
          # X is a matrix, while the likelyhoods are a vector for each word which corresponds exactly with the matrix in the same order
          # So in the matrix each word weight gets mutiplied by the probabilty of the logPrior 
          # So we are executing here sum(log(P(xi | C)) x TFIDF(xi))
          #            uber eats
          # uber      | 0    1 | x uber |0.1|
          # uber eats | 1    1 |   eats |0.2|
          # The vector is the likelyhoods and the matrix is X 
          # Out of this we get 
          # |7| uber 
          # |8| uber eats
          dotProduct = numpy.dot(X, self.logLikelyhoods[category])
          # Now we add the prior to our vector from the dot product 
          #     (|7|) uber 
          # 7 + (|8|) uber eats
          # The 7 is added to the entire vector here 
          scoreCategory = numpy.add(prior, dotProduct)
          # Here each score vector is added like [[14,15],[100,200]]
          scores.append(scoreCategory)
       
       # We convert the scores nested array to instead have each array stored vertically and not horizontally 
       scores = numpy.column_stack(scores)
       # Then we can get the index of the category for each word horizontally (axis = 1 means compare horizontally)
       # Comparing horizontally means compare the value that the model computes for every single category for each transaction 
       indicesOfCategories = numpy.argmax(scores, axis=1)
       
       # Numpy supports fancy indexing, so if we pass a list of indices into a list, numpy will give us a new list 
       # corresponding to the values of the indices  
       # Because the score was calculated in correspondce with the categories list, ie we looped through all categories in the order of the 
       # uniqueCategories list, the index of the vectors comparing horizonally, is one to one and in order of the uniqueCategories list
       # So we can do the following
       winningCategories = self.uniqueCatergories[indicesOfCategories]
       
       # The winning categories in order of the transactions (the order of the transactions in the matrix X)
       return winningCategories
        

        
           



        

   






