import numpy as np 
import mpmath
from cluster_module import *


"""
	GMM clustering 
	Expectation maximization algorithm implementation
	author: Rafael Leon
	date: 2018/03/29
	standard: PEP-8
"""

"""
	Function: expectation_maximization
	Reads a file with coordinates parses them and returns a list of (x,y) coordinate tuples

	@params:
	-point_list: (list[(tuple(x,y))]) list of points 
	-gaussians: list(dictionary) list of random generated gaussian dictionaries
	-iteration_limit: (int) maximum iteratiuon count

	@returns:
	-coordinates: list of coordinate tuples [(x_i,y_i)]
"""

def expectation_maximization( point_list, gaussians, iteration_limit = 1000 ) :
	#print("POINT LIST: ", point_list)
	#E STEP
	i = 0
	delta_mu_x = 1
	delta_mu_y = 1
	delta_sigma = 1 
	#delta_pi = 1
	while i <= iteration_limit and (np.mean(delta_mu_y) > 1e-3 and np.mean(delta_mu_x) > 1e-3) and np.mean(delta_sigma) > 1e-3:
		delta_mu_x = []
		delta_mu_y = []
		delta_sigma = []
		#delta_pi = []
		print("iter: ", i)
		for cluster in gaussians :
				cluster['normalized_probs'] = []
		for point in point_list :

		
			R = 0.0
			for cluster in gaussians :
				e_ij = float(non_normalized_probability(cluster['pi'], 
												  2, 
												  cluster['sigma'],
												  point,
												  cluster['mu']))
				
				R += e_ij
				
			#print("R: ",R)
			
			for cluster in gaussians :
				e_ij = float(non_normalized_probability(cluster['pi'], 
												  2, 
												  cluster['sigma'],
												  point,
												  cluster['mu']))
				#sprint(e_ij)
				norm_e_ij = e_ij/(R + 0.0001)
				temp = cluster['normalized_probs']
				#print(len(temp))
				temp.append(norm_e_ij)
				cluster['normalized_probs'] = temp
				#print(len(cluster['normalized_probs']))

		#M step
		for cluster in gaussians :
			new_pi = update_pi(point_list, cluster)
			#old_pi = cluster['pi']
			cluster['pi'] = new_pi
			#delta_pi_i = abs(new_pi - old_pi)
			
			#delta_pi.append(delta_pi_i)

			
			new_mu = update_mu(point_list, cluster)
			old_mu = cluster['mu']
			cluster['mu'] = new_mu
			delta_mu_i = abs(new_mu - old_mu)
			delta_mu_x.append(delta_mu_i[0])
			delta_mu_y.append(delta_mu_i[1]) 
			#print(delta_mu_i)
			new_sigma = update_sigma(point_list, cluster)
			old_sigma = cluster['sigma']
			delta_sigma.append(abs(np.linalg.det(new_sigma) - np.linalg.det(old_sigma)))
			cluster['sigma'] = new_sigma
			
		# print("PI stability: ", np.mean(delta_pi))
		# print(delta_pi)
		#print("mu stability: ", np.mean(delta_mu))
		#print(delta_mu)
		i += 1


"""
	Function: non_normalized probs
	Calculates the non normalized probability for a point

	@params:
	pid = pi value of the gaussian function
	n = dimensions
	sigma = sigma matrix for the gaussian
	x = point to calculate
	mu= mean of the gaussian

	@returns:
	-e: probability for the point to be in the gaussian (NON NORMALIZED)
"""
def non_normalized_probability(pid, N, sigma, x, mu) :
	
	#exp_part = mpmath.exp(-0.5 * np.transpose(x - mu) @ np.linalg.inv(sigma) @ (x - mu))
	#print( exp_part )

	e = pid * ((2 * np.pi)**(-N/2)) * (abs(np.linalg.det(sigma)))**(-1/2) * mpmath.exp(-0.5 * (np.transpose(x - mu) @ np.linalg.inv(sigma) @ (x - mu)))
	#print(e)
	return e 

"""
	Function: update_pi
	calculates the new value for pi

	@params:
	-point_list: (list[(np.arrays(x,y))]) list of points 
	-gaussian: dictionary of gaussian

	@returns:
	-new_pi: new value of pi
"""
def update_pi(point_list, gaussian) :
	sum_e_ij = sum(gaussian['normalized_probs'])
	new_pi = sum_e_ij/len(point_list)
	
	return new_pi
"""
	Function: update_mu
	calculates the new value for mu

	@params:
	-point_list: (list[(np.arrays(x,y))]) list of points 
	-gaussian: dictionary of gaussian

	@returns:
	-new_mu: new value of mu (np.array)
"""
def update_mu(point_list, gaussian) : 
	sum_e_ij = sum(gaussian['normalized_probs'])
	numerator = 0
	norm_prob_list = gaussian['normalized_probs']
	for i in range(0,len(point_list)) :
		numerator += point_list[i]*norm_prob_list[i]

	new_mu = numerator/sum_e_ij
	#new_mu = np.array([float(new_mu[0]),float(new_mu[1])])
	return new_mu

"""
	Function: update_sigma
	calculates the new value for sigma matrix

	@params:
	-point_list: (list[(np.arrays(x,y))]) list of points 
	-gaussian: dictionary of gaussian

	@returns:
	-new_sigma: new value of sigma matrix (np.array)
"""
def update_sigma(point_list, gaussian) :
	sum_e_ij = sum(gaussian['normalized_probs'])
	#print(sum_e_ij)
	numerator = 0
	norm_prob_list = gaussian['normalized_probs']
	#print("NORMMSSS: ", norm_prob_list)
	mu = gaussian['mu']
	#print(mu)
	for i in range(0, len(point_list)) :
		x = point_list[i]
		#print( np.transpose(np.array([[x-mu]])) @ np.array([[x-mu]]))
		numerator = numerator + ((norm_prob_list[i] + 0.00001) * (np.transpose(np.array([x-mu])) @ np.array([x-mu])))

	new_sigma =  numerator/sum_e_ij
	if np.allclose(np.linalg.det(new_sigma), 0) :
		print("WHY THE HELL IS THIS SINGULAR\n", new_sigma)
	return new_sigma