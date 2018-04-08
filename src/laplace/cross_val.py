from training import *
from corpus_prep import *
from clasification import *

"""
	LAPLACE SMOOTHING SPAM-HAM
	Cross-validation execution for spam ham classifier 
	author: Rafael Leon
	date: 2018/03/29
	standard: PEP-8
"""
crossval_dataset = read_corpus("datasets/crossval.csv", "\t")
training_dataset = read_corpus("datasets/training.csv", "\t")

trainingHam = training_dataset[training_dataset.iloc[:,0] == "ham"]
trainingSpam = training_dataset[training_dataset.iloc[:,0] == "spam"]
training_dict, count_training = bag_of_words(training_dataset.iloc[:,1])
ham_dict, count_ham = bag_of_words(trainingHam.iloc[:,1])
spam_dict, count_spam = bag_of_words(trainingSpam.iloc[:,1])

trained_set = read_json("training_probs.json")

X = len(training_dict)

k = 1
min_E = 1
min_k = 0
crossval_classified = classify_messages(crossval_dataset.iloc[:,1].values, trained_set)
E = prediction_error(crossval_dataset.iloc[:,0].values, crossval_classified)
#print(E)
while min_E > 0.01 and k < 100: 
	#print(min_E)
	if min_E > E:
		min_E = E
		min_k = k

	k += 0.01
	ham_prob = laplace_estimator(len(trainingHam.iloc[:,1]), len(training_dataset.iloc[:,1]), k, 2)
	spam_prob = 1 - ham_prob
	ham_prob_dict = training_probability(ham_dict, k, count_ham, X)
	spam_prob_dict = training_probability(spam_dict, k, count_spam, X)

	probs = {'ham_prob': ham_prob,
			'spam_prob': spam_prob,
			'ham_prob_dict': ham_prob_dict,
			'spam_prob_dict': spam_prob_dict}
	crossval_classified = classify_messages(crossval_dataset.iloc[:,1].values, probs)
	
	E = prediction_error(crossval_dataset.iloc[:,0].values, crossval_classified)
	
	print("E: ", E)
	print("Min E: ", min_E)
	print("Min K: ", min_k)
	print("K: ", k)

#print(crossval_classified)
	