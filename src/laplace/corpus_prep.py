import numpy as np 
import pandas as pd 
import re
import json
from collections import defaultdict

"""
	LAPLACE SMOOTHING SPAM-HAM
	Corpus preparation library modules for Spam and Ham Clasifier
	author: Rafael Leon
	date: 2018/03/29
	standard: PEP-8
"""



"""
	Function: read_corpus
	Reads from a TXT and returns a pandas dataframe
	@params:
	-file: (string) file name
	-delimiter: (string) delimiter to split columns by. DEFAULT "\t"
	@returns:
	-pandas dataframe
"""
def read_corpus( file, delimiter = "\t" ) :
	dataset = pd.read_csv(file, sep = delimiter)
	return dataset
"""
	Function: write_json
	Reads from a TXT and returns a pandas dataframe
	@params:
	-file: (string) file name
	-delimiter: (string) delimiter to split columns by. DEFAULT "\t"
	@returns:
	-None
"""
def write_json( file_name, dictionary) :
	with open("probabilities/"+file_name, 'w') as file:
		file.write(json.dumps(dictionary))

	return None

"""
	Function: write_json
	Reads from a TXT and returns a pandas dataframe
	@params:
	-file: (string) file name
	-delimiter: (string) delimiter to split columns by. DEFAULT "\t"
	@returns:
	-None
"""
def read_json( file_name) :
	with open("probabilities/"+file_name, 'r') as file:
		prob_dict = json.load(file)

	return prob_dict


"""
	Function: data_set_splitter
	Splits the dataframe in 3 sections for training, crossvalidation and testing
	@params:
	-dataset: (pandas.DataFrame) the data set to split
	-training_set_size (float): number betweeen 0 and 1
	-cross_val_size (float): number between 0 and 1

	@returns:
	-pandas dataframe x3, training, test and crossval 
"""
def data_set_splitter(dataset, training_set_size, cross_val_size) :
	if training_set_size + cross_val_size > 1.0 : 
		print("Training set size + crossvalidation set size must be lower than 1.00")
		return None, None, None
	else :
		test_set_lower_bound = training_set_size + cross_val_size
		training_set = []
		cross_val_set = []
		test_set = []
		for index, row in dataset.iterrows() :
			uniform_sample = np.random.uniform()
			i = list(row)
			if uniform_sample < training_set_size : 
				training_set.append(i)
			elif uniform_sample >= training_set_size and uniform_sample < test_set_lower_bound :
				cross_val_set.append(i)
			else :
				test_set.append(i)

		training_set = pd.DataFrame(training_set)
		cross_val_set = pd.DataFrame(cross_val_set)
		test_set = pd.DataFrame(test_set)

		return training_set, cross_val_set, test_set	




"""
	Function: string_sanitizer
	Sanitizes a string using regular expresions
	@params:
	-dirty_str: (string) String to sanitize

	@returns:
	-sanitized string
"""
def string_sanitizer( dirty_str ) :
	sanitized = re.sub(r'"|#|&|�|"$"|-|\'|!|\.|,|\*|\+|\^|�|@|�|�|=|>|<|;|\x92|\x93|\x94|\?|\(|\)|\[|\]|\}|\{|%' ,
					'',dirty_str, flags=re.IGNORECASE)
	sanitized = sanitized.lower()
	return sanitized


"""
	Function: bag_of_words
	Makes a bag of words from an array of strings
	@params:
	-string_array: (array) Array of strings to turn into a bag of words
	@returns:
	-bag_of_words: dictionary with word:frequency
	-count: observation count in dictionary
"""
def bag_of_words( string_array ) :
	d = defaultdict(int)
	count = 0
	for text in string_array :
		split_text = text.split()
		for word in split_text:
			d[word] += 1
			count += 1
	return d, count

