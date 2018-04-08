from input_prep import *
from cluster_module import *
from expectation_maximization import *
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pylab as p

"""
	GMM clustering 
	Main execution after having generated the k gaussians
	Classifies a single point at a time using GMM clusters
	author: Rafael Leon
	date: 2018/03/29
	standard: PEP-8
"""

gaussians = np.load("gaussians/clusters.npy")
#print(gaussians)
menu = True
while menu :
	print("-------------------------------------------------")
	print('WELCOME TO THE GAUSSIAN MIXED MODEL CLASSIFIER')
	print("-------------------------------------------------")
	print('select the number of the option you want')
	print("-------------------------------------------------")
	print('1. Classify a point')
	print('2. Exit')
	option = input("")
	if option == "1":		
		to_classify = input("Write the point to classify: ")

		point = coordinate_parser(to_classify)

		probs = []
		for cluster in gaussians:
			e_ij = float(non_normalized_probability(cluster['pi'], 
														  2, 
														  cluster['sigma'],
														  point,
														  cluster['mu']))
			probs.append(e_ij)

		cluster_number = probs.index(max(probs))
		print("-------------------------------------------------")
		print("Classified to cluster: \n#", cluster_number+1)
		print("-------------------------------------------------")

	elif option == "2" :
		menu = False
		print("-------------------------------------------------")
		print("BYE")
		print("-------------------------------------------------")
	else:
		print("-------------------------------------------------")
		print("ERROR: invalid option!")
		print("-------------------------------------------------")