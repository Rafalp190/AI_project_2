import numpy as np 
import pandas as pd 

from corpus_prep import *

"""
	LAPLACE SMOOTHING SPAM-HAM
	training library modules for Spam and Ham Clasifier
	author: Rafael Leon
	date: 2018/03/29
	standard: PEP-8
"""

dataset = read_corpus("probabilities/test_corpus.txt", "\t")


for i in range(len(dataset.iloc[:,1])):
	clean = string_sanitizer(dataset.iloc[i,1])
	dataset.iloc[i,1] = clean

training, crossval, test = data_set_splitter(dataset, 0.80, 0.10)

training.to_csv("datasets/training.csv", sep="\t", index=False)
crossval.to_csv("datasets/crossval.csv", sep="\t", index=False)
test.to_csv("datasets/test.csv", sep="\t", index=False)

#Splits the dataframe in arrays of Ham and spam
trainingHam = training[training.iloc[:,0] == "ham"]
#print(trainingHam.describe())
trainingSpam = training[training.iloc[:,0] == "spam"]
#print(trainingSpam.describe())

#Calculates spam and ham bag of words
ham_dict, count_ham = bag_of_words(trainingHam.iloc[:,1])
spam_dict, count_spam = bag_of_words(trainingSpam.iloc[:,1])
training_dict, count_training = bag_of_words(training.iloc[:,1])



#--------------------------------------------------------------------
# Bayesian Classifier Variables for a specific k
k = 7
X = len(training_dict)

ham_prob = laplace_estimator(len(trainingHam.iloc[:,1]), len(training.iloc[:,1]), k, 2)
spam_prob = 1 - ham_prob
ham_prob_dict = training_probability(ham_dict, k, count_ham, X)
spam_prob_dict = training_probability(spam_dict, k, count_spam, X)

probs = {'ham_prob': ham_prob,
		'spam_prob': spam_prob,
		'ham_prob_dict': ham_prob_dict,
		'spam_prob_dict': spam_prob_dict}

write_json("training_probs.json", probs)
