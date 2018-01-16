
# coding: utf-8

# In[1]:


import numpy as np
import scipy.special
import MNISTLoader
#from pkl, retrieve MNIST


# In[2]:


class Network:
    def __init__(self,neurons):
        self.layers=len(neurons)
        self.neurons=neurons
        
        self.weights=[]
        self.biases=[]
        
    #if 3 layers then i ranges from 1 to 2 
    #initialized weights
        for i in range(1,self.layers):
            #layer_weight=None
            rows=neurons[i]
            cols=neurons[i-1]
            #layer_weight=np.random.randn(rows,cols)
            layer_weight=np.zeros((rows,cols))
            #random.random will give uniform distribution and not random
            self.weights.append(layer_weight)
            
            #layer_biases initializing
            #layer_bias=np.random.randn(rows,1)
            layer_bias=np.zeros((rows,1))
            self.biases.append(layer_bias)
            


# In[3]:


train_data,valid_data,test_data=MNISTLoader.load_data_wrapper()
#im=train_data[0].reshape(28,28)


# In[4]:


def feedforward(my_net,X,Inputs,Wt_Avg):
 
    #Inputs.append(X)   
    for i in range (1,my_net.layers):
        W = my_net.weights[i-1]
        B = my_net.biases[i-1]
        if (i==1):
            Z = np.dot(W,X) + B
            Inputs.append(X)
        else:
            Z = np.dot(W,Inputs[-1]) + B
            
        Wt_Avg.append(Z)
        O = sigmoid(Z)
        Inputs.append(O)#Stores Activation
    #return karna hai kya???
    #return Inputs


# In[5]:


def backprop(my_net,Wt_Avg,Delta):
    
            delta=Delta[0]
            for i in range(my_net.layers-1,1,-1):
                #print(i)
                wt_avg=Wt_Avg[i-2]#************************
                wd=np.dot(my_net.weights[i-1].transpose(),delta)
                sd=sigmoid_ddx(wt_avg)
                delta = wd*sd
                Delta.append(delta)
    #return delta


# In[6]:


def Update(Inputs,Delta,my_net,alpha):
    
    #IF NO OF LAYERS IS 5 THEN
    #THE NUMBER OF ENTRIES IN INPUTS IS 5
    #THE NUMBER OF ENTRIES IN DELTA IS 4
 
    for i in range(my_net.layers-1):
        dcdw = (Delta[i]).dot(Inputs[i].transpose())
        #dcdw = np.average(dcdw,axis=1)
        my_net.weights[i] -= alpha*dcdw
        #print(dcdw.shape)
        dcb = np.average(Delta[i],axis=1).reshape(Delta[i].shape[0],1)
        my_net.biases[i] -= alpha*dcb
        
        #print("dcdw for layer {0} is {1}".format(i,dcdw))
        #print(mynet
    return


# In[7]:


def sigmoid(z):
    return 1/(1.0+scipy.special.expit(-z))


# In[8]:


def sigmoid_ddx(z):
    return sigmoid(z)*(1-sigmoid(z))


# In[9]:


def cost_gradient(activation,label):
    return activation-label


# In[10]:


#def test(Output,Target):#Error wala not cost
#    #only output is of the dimension (10,No of training samples)
#    
#    Prediction=np.argmax(Output,axis=0)#+1
#    Accuracy=(sum((Prediction==Target).astype(np.float32))/Target.shape[0])*100
#    return Accuracy


# In[11]:


def test(my_net,data,Inputs,Wt_Avg,Delta,alpha):#Error wala not cost
    #only output is of the dimension (10,No of training samples)
    feedforward(my_net,data[0],Inputs,Wt_Avg)
    Prediction=np.argmax(Inputs[-1],axis=0)#+1
    Accuracy=(sum((Prediction==data[1]).astype(np.float32))/data[1].shape[0])*100
    return Accuracy


# In[12]:


def train_GD(my_net,train_data,validation_data,epochs=50):
    for i in range (epochs):
        Inputs=[]
        Wt_Avg=[]
        Delta=[]
        feedforward(my_net,train_data[0],Inputs,Wt_Avg)
        final_del = cost_gradient(Inputs[-1],train_data[1])*sigmoid_ddx(Wt_Avg[-1])
        Delta.append(final_del)
        backprop(my_net,Wt_Avg,Delta)
        Delta = Delta[::-1]
        Update(Inputs,Delta,my_net,1)
        accuracy=test(my_net,validation_data,Inputs,Wt_Avg,Delta,1)

        print("Epoch:",i,"Accuracy",accuracy)
    return


# my_net=Network([784,16,10])
# X=train_data[0]
# #X=valid_data[0] #Used for testing data
# #Stores Activation
# Inputs=[]
# #Store Weighted Average
# Wt_Avg=[]
# feedforward(my_net,X,Inputs,Wt_Avg)
# #Stores all deltas in reverse order ig
# Delta=[]
# 
# Delta.append(final_del)
# backprop(my_net,Wt_Avg,Delta)
# #Delta=list(reversed(Delta))
# Delta = Delta[::-1]
# #print(Delta)
# Update(Inputs,Delta,my_net,1)
# for i in range (2):
#     print(my_net.weights[i],)
#     #print()
# #print(Inputs[])
# #Inputs[-1].shape
# Target=valid_data[1]
# #test(Inputs[-1],Target)

# In[13]:


#test(my_net,data,Inputs,Wt_Avg,Delta,alpha)


# In[14]:


my_net=Network([784,16,10])
X=train_data[0]
train_GD(my_net,train_data,valid_data,epochs=50)

