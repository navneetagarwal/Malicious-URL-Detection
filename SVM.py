import sys
import os
# Write the path to the python file in your libsvm directory to  import from svmutil.
sys.path.append('libsvm-3.21/python')
from svmutil import *

y,x = svm_read_problem('Day0.svm')
svm_problem_instance = svm_problem(y[:8000],x[:8000]) # loading the training data

'''
options:
-s svm_type : set type of SVM (default 0)
	0 -- C-SVC		(multi-class classification)
	1 -- nu-SVC		(multi-class classification)
	2 -- one-class SVM	
	3 -- epsilon-SVR	(regression)
	4 -- nu-SVR		(regression)
-t kernel_type : set type of kernel function (default 2)
	0 -- linear: u'*v
	1 -- polynomial: (gamma*u'*v + coef0)^degree
	2 -- radial basis function: exp(-gamma*|u-v|^2)
	3 -- sigmoid: tanh(gamma*u'*v + coef0)
	4 -- precomputed kernel (kernel values in training_set_file)
-d degree : set degree in kernel function (default 3)
-g gamma : set gamma in kernel function (default 1/num_features)
-r coef0 : set coef0 in kernel function (default 0)
-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
-m cachesize : set cache memory size in MB (default 100)
-e epsilon : set tolerance of termination criterion (default 0.001)
-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
-v n: n-fold cross validation mode
-q : quiet mode (no outputs)

'''
svm_param = svm_parameter('-t 3 -c 10000  -h 0') # kernel is sigmoid, cost/C is 10000 
model = svm_train(svm_problem_instance, svm_param) # trainin the model
prediction_labels, prediction_accuracy, prediction_vals = svm_predict(y[8000:],x[8000:],model) #predicting on test data
print 'Model Prediction Accuracy on 8000 datapoints of test data: {0}'.format(prediction_accuracy) # printint accuracy
print 'Perceptron Model'
os.system('python Dimentionality_Reduction.py')
os.system('python Perceptron.py')
print 'NeuralNetwork Model'
os.system('python NeuralNets_comment.py')

