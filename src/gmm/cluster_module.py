import numpy as np 

"""
	GMM clustering 
	Cluster module functions to build the expectation_maximization algorithm

	author: Rafael Leon
	date: 2018/03/29
	standard: PEP-8
"""

"""
	Function: random_SIGMA
	creates a random 2x2 reversible matrix 
	@params:

	@returns:
	-random_mat: 2x2 reversible numpy array
"""
def random_SIGMA() :
	irreversible = True
	random_matrix = np.zeros((2,2))
	while irreversible :
		random_matrix = np.random.rand(2,2)
		if not np.allclose(np.linalg.det(random_matrix), 0) :
			irreversible = False
	return random_matrix
"""
	Function: random_U
	creates a random U tuple (x,y)
	@params:

	@returns:
	-random_tuple: np.array (x,y)
"""
def random_U() :
	random_array = np.random.rand(2)
	x = random_array[0]
	y = random_array[1]
	random_tuple = np.array([x,y])
	return 	random_tuple
"""
	Function: random_PI
	creates k random PI values  ∑πi = 1 y πi > 0
	@params:
	k: Gaussian count
	@returns:
	-PI_list: list of k random PI values ∑πi = 1 y πi > 0
"""
def random_PI(k) :
	PI_list = []
	sum_PI = 0
	for i in range( 0, k-1) :
		random_pi = np.random.uniform(sum_PI,1)
		#print(random_pi)
		PI_list.append(1-random_pi)
		sum_PI = sum(PI_list)
		#print(sum_PI)
	PI_list.append(1-sum_PI)
	#print(sum(PI_list))
	return PI_list

"""
	Function: random_GAUSSIANS
	creates k random gaussians
	@params:
	k: Gaussian count
	@returns:
	-gaussians: list of k random gaussian dictionaries 
"""
def random_GAUSSIANS(k) :

	random_pies = random_PI(k)
	gaussians = []
	for i in range(0,k) :
		mu = random_U()
		sigma = random_SIGMA()

		gaussian = {"mu": mu,
					"sigma": sigma,
					"pi": random_pies[i],
					"normalized_probs": []}
		gaussians.append(gaussian)

	return gaussians