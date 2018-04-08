import numpy as np 
import json
"""
	GMM clustering 
	File reading and input prep for coordinates in the format "[x,y]"
	author: Rafael Leon
	date: 2018/03/29
	standard: PEP-8
"""

"""
	Function: coordinate_file_reader
	Reads a file with coordinates parses them and returns a list of (x,y) coordinate tuples

	@params:
	-file:(string) directory of file

	@returns:
	-coordinates: list of coordinate tuples [(x_i,y_i)]
"""
def coordinate_file_reader(file):

	file = open(file, "r")
	lines = file.readlines()
	file.close() 

	coordinates = list(map(coordinate_parser, lines))

	return coordinates

"""
	Function: coordinate_parser
	Reads a string with coordinates parses them and returns an (x,y) coordinate
	@params:
	-file:(string) coordinate string "[x,y]\n"

	@returns:
	-coordinate:  coordinate np.array (x,y)
"""

def coordinate_parser(coordinate_string):
	coordinate_string = coordinate_string.strip('\n')
	coordinate_string = coordinate_string[1:-1]
	coordinates = coordinate_string.split(",")
	x = float(coordinates[0])
	y = float(coordinates[1])

	return np.array([x,y])


