import numpy as np
import math
import csv
import sys

def __sigmoid__(x):
    return 1.0/(1.0+np.exp(-x))
def __sigmoid_prime__(x):
    return __sigmoid__(x)*(1-__sigmoid__(x)) #derivative of sigmoid wrt x

def __tanh__(x):
    return np.tanh(x)
def __tanh_prime__(x):
    return 1-np.tanh(x)**2 #derivative of tanh wrt x

class NeuralNetwork:
    def __init__(self, layers, activation='sigmoid',feature_size=57):
        if activation =='sigmoid':
            self.activation = __sigmoid__
            self.activation_prime = __sigmoid_prime__
        elif activation == 'tanh':
            self.activation = __tanh__
            self.activation_prime = __tanh_prime__

        self.weights=[]         #Weight Wij = weght for jth node in l layer, for ith node in l-1 layer
        layers[0]=layers[0]+1   #one bias node
        for i in range(len(layers)-1):
            r=2*np.random.random((layers[i],layers[i+1]))-1
            #r[0][:]=0            # setting weights for bias to 0
            self.weights.append(r)
    def __train__(self, X, y, learning_rate=0.05, epochs=10000,lambda_param=0.0001):
        ones = np.atleast_2d(np.ones(X.shape[0]))        #adding bias to the feature vectors
        X = np.concatenate((ones.T,X),axis=1)

        for k in range(epochs): #iterating over the dataset
            i=np.random.randint(X.shape[0])
            activation = [X[i]]
            
            #forward propogation
            for l in range(len(self.weights)):
                dot_product = np.dot(activation[l],self.weights[l]) #vector with jth element: <sigma(l-1),Wj>
                activation.append(self.activation(dot_product))

            #  Important equations:
            #  delta(L)=(a(L)-y)*a'(L)
            #  delta(l)=(delta(l+1).weight(l+1))*a'(l)
            #  grad_ij(l)=a_i(l-1)delta_j(l)
            
            error= activation[-1]-y[i]  # derivative of squared error at the last layer
            deltas = [error*self.activation_prime(activation[-1])] #=delta(L)
            for l in range(len(self.weights)-2,-1,-1):  # computing delta(l) layer l (all units)
                delta = self.weights[l+1].dot(deltas[-1])*self.activation_prime(activation[l+1]) #=delta(l)
                deltas.append(delta)
            deltas.reverse()

            for i in range(len(self.weights)): # updating the weights for each layer 
                layer = np.atleast_2d(activation[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] -= learning_rate*layer.T.dot(delta) #+lambda_param*self.weights[i]

            if (k+1)%1000 ==0:
                print 'epoch',k+1

    def __predict__(self,x):
        x=np.concatenate((np.ones(1).T,x),axis=1) #for the bias
        for l in range(0,len(self.weights)): #forward propagation
            x= self.activation(x.dot(self.weights[l]))
        return x
    

if __name__ == '__main__':
    #getting the dataset from Train.csv file(taking only 16000 points in total[train + test])
    raw_data = open('svm_to_csv.csv')
    dataset = np.loadtxt(raw_data,delimiter=',')
    X_train = dataset[:8000,1:76]
    y_train = dataset[:8000,:1]

    #getting the Neural Network with suitable architecture: hidden unit per layer= 4*#inputs
    hidden_units=X_train.shape[1];
    N = NeuralNetwork([hidden_units,4*hidden_units,4*hidden_units,4*hidden_units,1],activation='sigmoid')
    N.__train__(X_train,y_train,epochs=25000)

    X_test=dataset[8000:16000,1:76]
    y_test=dataset[8000:16000,:1]
    prediction = np.array([ [0] if N.__predict__(x) <0.5 else [1] for x in X_test ])
    error = np.sum((prediction-y_test)**2)
    #print N.weights
    print "Number of points missclassified in {0} test points is: {1}".format(8000,error)
    













    
    '''
    test_data=open('TestX.csv')
    dataset=np.loadtxt(test_data,delimiter=',')
    X=dataset[:,:54]
    prediction = [ 0 if N.predict(x)<0.5 else 1 for x in X]
    print prediction
    prediction = [ [str(i),str(prediction[i])] for i in range(len(prediction))]
            
    print "weights",N.weights
    i=1
    try:
        with open("TestY.csv",'wb') as CSVfile:
            CSVWriter = csv.writer(CSVfile)
            CSVWriter.writerow(['Id','Label'])
            for row in prediction:
                CSVWriter.writerow(row)
    except(IOError) as e:
        print("Error in opening file!{0}!",format(e.strerror))
    except:
        print(sys.exc_info()[0])
   '''
