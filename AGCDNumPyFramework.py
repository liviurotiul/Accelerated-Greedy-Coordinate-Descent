import numpy as np
import math
import matplotlib.pyplot as plt

ITERATIONS = 130000
NO_OF_DIMENSIONS_G = 3
FUNCTIONS = 4

def test_function(func):
	X = Z = np.random.randn((function_dims(func)))  * 400
	plt.plot(AGCD(func, X, Z), label="AGCD")
	print("________________________")
	plt.plot(ASCD(func, X, Z), label="ASCD")
	print("________________________")
	plt.plot(ARCD(func, X, Z), label="ARCD")
	plt.legend(loc="upper right")

	plt.show()


def param(f):
	if f == 0:
		return 0.1
	if f == 1:
		return 0.00014
	if f == 2:
		return 0.08
	if f == 3:
		return 0.000000005

def function_dims(f):
	if f == 0 or f == 1:
			return NO_OF_DIMENSIONS_G
	if f == 2:
			return 2
	if f == 3:
			return 3

def grad(x_L, function, dimension=None):
	X = x_L[dimension]
	if function == 0: # for sum of squares function
		return 2*X
	if function == 1: # for (x1*x2*...*xn)^2
		return 2*np.prod(x_L**2)/x_L[dimension]
	if function == 2: # for (x1^2+sinx2)
		if dimension == 0:
			return x_L[0]*2
		return math.cos(x_L[1])
	if function == 3:
		if dimension == 0:
			return (x_L[0]*x_L[2] + x_L[1]*x_L[2])
		if dimension == 1:
			return (x_L[2]*x_L[1] + x_L[0] + 5*x_L[1] - x_L[0]*x_L[2])
		if dimension == 2:
			return x_L[2] *( x_L[2] + 0.1*x_L[1])
			

def f_eval(x_L, function):
	
	if function == 0: #for sum of squares function
		return np.sum(np.asarray([t**2 for t in x_L]))
	if function == 1: # for (x1*x2*...*xn)^2
		return np.prod(x_L**2)
	if function == 2:
		return x_L[0]**2 + math.sin(x_L[0]) + x_L[1] + math.sin(x_L[0])
	if function == 3:
		return np.sum(np.asarray([t**2 for t in x_L]))

def update_THETA(theta):
	temp = 1/(theta**2)
	return (-temp + math.sqrt(temp**2 + 4*temp))/2



def ARCD(function, X, Z):
	values = []
	THETA = 1
	NO_OF_DIMENSIONS = function_dims(function)
	x, z = X, Z
	for index in range(ITERATIONS):
		j1 = np.random.randint(NO_OF_DIMENSIONS) #I replaced j1 and j2 with i and j respectively for convenience
		j2 = np.random.randint(NO_OF_DIMENSIONS)
		y = (1-THETA)*x + THETA*z
		temp = np.zeros(NO_OF_DIMENSIONS)
		temp[j1] = param(function)*grad(x, function=function, dimension=j1)
		x = y - temp
		temp = np.zeros(NO_OF_DIMENSIONS)
		temp[j2] = param(function)*grad(z, function=function, dimension=j2)
		z = z - temp
		print(y)
		values.append(f_eval(y,function))
		THETA = update_THETA(THETA)

	return values

def AGCD(function, X, Z):
	values = []
	THETA = 1
	NO_OF_DIMENSIONS = function_dims(function)
	x, z = X, Z
	for index in range(ITERATIONS):
		y = (1-THETA)*x + THETA*z
		j1 = j2 = np.argmax(np.asarray([grad(x_L=y, function=function,
										dimension=dim) for dim,temp in enumerate(y)])) #I replaced j1 and j2 with i and j respectively for convenience	
		temp = np.zeros(NO_OF_DIMENSIONS)
		# print(param(function))
		temp[j1] = param(function)*grad(x, function=function, dimension=j1)
		x = y - temp
		temp = np.zeros(NO_OF_DIMENSIONS)

		temp[j2] = param(function)*grad(z, function=function, dimension=j2)
		z = z - temp
		print(y)
		values.append(f_eval(y,function))
		THETA = update_THETA(THETA)
	return values

def ASCD(function, X, Z):
	values = []
	THETA = 1
	NO_OF_DIMENSIONS = function_dims(function)
	x, z = X, Z
	for index in range(ITERATIONS):
		y = (1-THETA)*x + THETA*z
		j1 = np.argmax(np.asarray([grad(x_L=y, function=function,
										dimension=dim) for dim,temp in enumerate(y)])) #I replaced j1 and j2 with i and j respectively for convenience	
		j2 = np.random.randint(NO_OF_DIMENSIONS)
		temp = np.zeros(NO_OF_DIMENSIONS)
		temp[j1] = param(function)*grad(x, function=function, dimension=j1)
		x = y - temp
		temp = np.zeros(NO_OF_DIMENSIONS)
		temp[j2] = param(function)*grad(z, function=function, dimension=j2)
		z = z - temp
		values.append(f_eval(y,function))
		print(y)
		THETA = update_THETA(THETA)
	return values



# test_function(0)

test_function(3) 