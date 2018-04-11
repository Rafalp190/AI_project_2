from corpus_prep import *
from training import *
from clasification import *



training_dataset = read_corpus("datasets/training.csv", "\t")
test_dataset = read_corpus("datasets/test_sms.txt", "\t")

for i in range(len(test_dataset.iloc[:,1])):
	clean = string_sanitizer(test_dataset.iloc[i,1])
	test_dataset.iloc[i,1] = clean

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

messages = test_dataset.iloc[:,1].values 

#E = prediction_error(test_dataset.iloc[:,0].values, test_classified)
out_df = pd.DataFrame({'classification': test_classified, 'message': messages})
print('-----------------------------------------------')
print('CLASSIFICATION OF SET OF MESSAGES')
print('The classification was completed successfully')
print('-----------------------------------------------')
print('Information :')
print('K :', k)
#print("Prediction Error:", E)
print("Total Messages: ", len(test_classified))
#print("Incorrectly Classified: ", int(len(test_classified)*E))
print('-----------------------------------------------')
print(out_df)
print('The complete message output was stored in output/test_classified.txt')

out_df.to_csv("output/test_classified.txt", sep="\t", index=False)
