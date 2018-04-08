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
	Creates the gaussians and shows a plot for the contours and the points
	author: Rafael Leon
	date: 2018/03/29
	standard: PEP-8
"""

coords = coordinate_file_reader("coordinates/coordinates.txt")

gaussians = random_GAUSSIANS(3)
expectation_maximization(coords, gaussians, 1000)
np.save("gaussians/clusters", gaussians)
#print(gaussians)
x = np.arange(-500.0, 500.0, 1)
y = np.arange(-500.0, 500.0, 1)
X, Y = np.meshgrid(x, y)

for i in gaussians:
	sigma = i['sigma']
	mu = i['mu']
	Z = mlab.bivariate_normal(X, Y, sigma[0,0], sigma[1,1], mu[0], mu[1], sigma[0,1])

	CS = plt.contour(X, Y, Z)


	#plt.contour(Z, vmin=min(Z), vmax=max(Z))
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Gaussian Plots')
x, y = np.asarray(coords).T
plt.scatter(x,y)
plt.show()
