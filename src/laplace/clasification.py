import numpy as np
from corpus_prep import *


"""
	Function: text_probability
	Calculates the probability for a text to be spam or ham
	@params:
	-text: (string) string to calculate prob of
	-prob_dict (dictionary): dictionary of probabilities
	@returns:
	-pro_ham, prob_spam: the probability of the text to belong to ham or spam
"""
def text_probability(text, prob_dict) : 
	ham_prob = prob_dict['ham_prob']
	spam_prob = prob_dict['spam_prob']
	ham_prob_dict = prob_dict['ham_prob_dict']
	spam_prob_dict = prob_dict['spam_prob_dict']

	text_list = text.split()
	ham_prob_list = []
	spam_prob_list = []
	for i in text_list:
		if i in ham_prob_dict :
			ham_prob_list.append(ham_prob_dict[i])
		else :
			ham_prob_list.append(ham_prob_dict['not_known'])
		
		if i in spam_prob_dict :
			spam_prob_list.append(spam_prob_dict[i])
		else :
			spam_prob_list.append(spam_prob_dict['not_known'])

	prob_ham = 1.0
	for p in ham_prob_list :
		prob_ham = prob_ham*p

	prob_ham = ham_prob * prob_ham



	prob_spam = 1.0
	for p in spam_prob_list :
		prob_spam = prob_spam*p

	prob_spam = spam_prob * prob_spam

	return prob_ham, prob_spam
"""
	Function: classify
	Classifies between spam or ham
	@params:
	-prob_list: the probabilities for spam and ham
	@returns:
	-string: spam or ham respectively
"""
def classify(prob_list):
	if prob_list[0] > prob_list[1]:
		return "ham"
	else : 
		return "spam"


"""
	Function: classify_messages
	Classifies between spam or ham
	@params:
	-message_array: the array of messages to classify
	-prob_dict: the probabilities for spam ham and the dictionary for all the trained words
	@returns:
	-classified: array of classifications
"""
def classify_messages(message_array, prob_dict):
	classified  = []
	for m in message_array:
		ham_probability, spam_probability = text_probability(m, prob_dict)
		m_class = classify([ham_probability, spam_probability])
		classified.append(m_class)

	return classified 
"""
	Function: prediction_error
	Function to calculate the standard error
	@params:
	-theoretical(list): list of theoretical values of classification
	-experimental(list): list of experimental values of classification
	@returns:
	-error: float of the standard error
"""
def prediction_error(theoretical, experimental) :
	positives = 0
	for i in range(len(theoretical)):
		if theoretical[i] == experimental[i]:
			positives += 1
	error = abs(len(theoretical) - positives)/len(theoretical)
	return error