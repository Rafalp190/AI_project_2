from corpus_prep import *
from training import *
from clasification import *

def prediction_error(theoretical, experimental) :
	positives = 0
	for i in range(len(theoretical)):
		if theoretical[i] == experimental[i]:
			positives += 1
	error = abs(len(theoretical) - positives)/len(theoretical)
	return error

training_dataset = read_corpus("datasets/training.csv", "\t")
test_dataset = read_corpus("datasets/test.csv", "\t")

trainingHam = training_dataset[training_dataset.iloc[:,0] == "ham"]
trainingSpam = training_dataset[training_dataset.iloc[:,0] == "spam"]
training_dict, count_training = bag_of_words(training_dataset.iloc[:,1])
ham_dict, count_ham = bag_of_words(trainingHam.iloc[:,1])
spam_dict, count_spam = bag_of_words(trainingSpam.iloc[:,1])

X = len(training_dict)
k = 1.2
ham_prob = laplace_estimator(len(trainingHam.iloc[:,1]), len(training_dataset.iloc[:,1]), k, 2)
spam_prob = 1 - ham_prob
ham_prob_dict = training_probability(ham_dict, k, count_ham, X)
spam_prob_dict = training_probability(spam_dict, k, count_spam, X)

probs = {'ham_prob': ham_prob,
		'spam_prob': spam_prob,
		'ham_prob_dict': ham_prob_dict,
		'spam_prob_dict': spam_prob_dict}

test_classified = classify_messages(test_dataset.iloc[:,1].values, probs)
test_dataset.insert(2,"test",test_classified)

E = prediction_error(test_dataset.iloc[:,0].values, test_classified)
print(E)

test_dataset.to_csv("datasets/test_classified.csv", sep=",", index=False)