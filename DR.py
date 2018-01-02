import numpy as np


def weighted_sum(X,W,B):
	Y=X.dot(np.W)+B
	return Y

def forward_pass(X,network):
	W=network.weights[0]
	B=network.biases[0]
	Inputs.append[X]
	Y = weighted_sum(X,W,B)
	Weighted_sums.append(Y)
	O=sigmoid(Y)
	for i in range (1,len(network.weights)-1): #Do While Types Hmm
	W=network.weights[i]
	B=network.biases[i]
	Inputs.append[O]
	Y = weighted_sum(O,W,B)
	Weighted_sums.append(Y)
	O=sigmoid(Y)
	O=np.argmax(O,axis=1)+1
	return O

def calc_cost(O,t):
	#AvgError

def backward_pass(network,alpha,Weighted_sums,Inputs):
	error=calc_error(Weighted_sums,O,t)
	#db=error
	#dw=[]
	#Updating Weights and Baises
	for i in range(len(error)):
		dw=Inputs[i].dot(error[i])
		db=error[i]
		network.weights[i]+=alpha*dw
		network.baises[i]+=alpha*db

def calc_error(O,t,Weighted_sums,do):
	do = -(t/O + (1-t)/(1-O))
	error=[]
	for i in range (len(Weighted_sums)):
	error[i] = do.dot(sigmoid_gradient(Weighted_sums[i]))
	return 0

def sigmoid_gradient(z):
	np.exp(-z)/(1.0+np.exp(-z)**2)


#Activation Function
def ReLu(z):
	return np.maximum(z,0,z)

def sigmoid(z):
		return 1/np.exp(-1*z)+1

class Network:
	def __init__(self,layers):
		if len(self.layer)<2:
			print("Cant Create NN w less than 2 layers")
			return
		self.layers = layers
		self.weights = []
		self.biases = [] #Couldve initialise but doesnt matter cause were gonna directtly append

		for i in range(1,len(layers)):
			rows=layers[i-1] #or use 0, len(layers)-1
			col = layers[i] #28
			layer_weight = np.random.random((row,cols))
			layer_bias = np.random.random((layers[i])) #28 random values [0.0,1.0)
			self.weights.append(layer_weight)
			self.baises.append(layer_bias)
			Inputs=[]
			Weighted_sums=[]
			alpha=0.25

#			my_net = Network([3,1])

#t=np.array([0.5,0.5,0.5,0.5,0.5])
#a=np.array([1,2,3,4])
#b=np.arrayy([1,1,1,1],[1,1,1,1],[1,1,1,1])
#a.dot(np.transpose(b)+baises