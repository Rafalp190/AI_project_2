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
"""
	Function: training_probabily
	calculates the probability dictionary for each word utilizing laplace smoothing

	@params:
	-bag_o_words:(dictionary) bag of words
	-k: (float) laplace smoothing factor
	-observations: (int) count of words in bag_o_words
	-X: number of different values for dependant variable
	@returns:
	-probability_dict: dictionary with the probability for each word
"""
def training_probability( bag_o_words, k, observations, X ) :
	probability_dict = dict()
	smooth_observations = float(observations + (k*X))
	for word in bag_o_words :
		word_freq = bag_o_words[word]
		word_observation = float(word_freq + k)
		probability = word_observation/smooth_observations
		probability_dict[word] = probability
	probability_dict['not_known'] = 0+float(k)/smooth_observations 
	return probability_dict


"""
	Function: laplace_estimator
	calculates the probability estimator for a K
	
	@params:
	-n_obs: (int) count of positive observations
	-n_total: (int) count of total observations
	-k: (float) laplace smoothing factor
	@returns:
	-laplacian estimator: value for the laplacian estimator for a specific K
"""
def laplace_estimator(n_obs, n_total, k, X) :
	n_obs = float(n_obs)
	n_total = float(n_total)
	k = float(k)
	estimator = (n_obs+k)/(n_total + (k*X))
	return estimator





