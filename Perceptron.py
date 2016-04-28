import numpy as np

''' Class to implement Perceptron
    using the simple perceptron update'''
class Perceptron:
    def __init__(self,feature_count=1):
        #creating the weight matrix
        self.weights = np.zeros(feature_count+1)
        self.weights[feature_count]=1.0
    def __train__(self,X,y,epochs=10000):
        # appending 1 for the bias 
        ones = np.atleast_2d(np.ones(X.shape[0])) 
        X=np.concatenate( (ones.T,X) , axis=1)
        #transforming the y from (0,1) to (-1,1) for the update
        y=np.array([1 if i>0 else -1 for i in y])

        counter=0 #keeps track of the number of iterations
        misclassified=True;

        #shuffling the dataset
        index=[i for i in range(X.shape[0])]
        index=np.random.permutation(index)
        X=X[index]
        
        while misclassified and counter<epochs:
            #finding a misclassified data point
            misclassified = False
            for i in range(len(X)):
                if(self.weights.dot(X[i])*y[i]<=0):
                    misclassified=True
                    break
            if misclassified:
                self.weights=self.weights+X[i]*y[i] #perceptron update

            counter=counter+1

            if counter%5000==0:
                print 'epochs:', counter
                #shuffling the dataset for stochastic update
                index=np.random.permutation(index)
                X=X[index]

    def __predict__(self,x):
        x=np.concatenate((np.ones(1),x),axis=1) #adding 1 for the bias
        return 1 if x.dot(self.weights)>0 else 0 #predicting the output

if __name__ == '__main__':
    #getting the dataset from file Train.csv in the folder
    raw_data = open('svm_to_csv.csv')
    dataset=np.loadtxt(raw_data,delimiter=',')
    X_train=dataset[:8000,1:1000]
    y_train=dataset[:8000,:1]

    #Creating the Percepptron object and training it
    P = Perceptron(X_train.shape[1])
    P.__train__(X_train,y_train,epochs=50000)

    #Testing the Perceptron model on the other half
    #of the dataset for cross validation
    X_test=dataset[8000:16000,1:1000]
    y_test=dataset[8000:16000,:1]
    prediction = np.array([ [P.__predict__(x)] for x in X_test])  #prediction
    
    error = np.sum((prediction-y_test)**2)  #finding the number of misclassified points
    print 'Number of Misclassified points in 8000 test points: ', error
