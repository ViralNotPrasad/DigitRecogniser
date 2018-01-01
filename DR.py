import numpy as np
def forward(X,W,B):
Y=X.dot(np.transpose(W)+B
O=sigmoid(Y)
	return O
def backward_pass():
	pass
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

#			my_net = Network([,])


a=np.array([1,2,3,4])
b=np.arrayy([1,1,1,1],[1,1,1,1],[1,1,1,1])
a.dot(np.transpose(b)+baises