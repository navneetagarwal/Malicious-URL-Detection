from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
import numpy as np
import scipy
from scipy import sparse

x = np.ones((3,4))
s = sparse.csr_matrix((3,4))
s[0,0] = 2
s[1,2] = 3
I,J = s.nonzero()
x[:] = 0
x[I,J] = s.data

# Loading data from Day0.svm in svm format
data1 = load_svmlight_file("Day0.svm")

# Opening the the file in f1
f1 = open('Day0.svm','r')

out = []

# If 1st character is +, then y = 1;
# Else y = 0;
for line in f1:
	if (line[0] == '+'):
		out.append(1)
	else :
		out.append(0)

f1.close()	# Closing the file

# Opening the file to write in.
f = open("svm_to_csv.csv",'w')

# For each line, Enter the values separated by a space.
for j in range(16000):

	f.write(str(out[j]) + ',')		# 1st character is y. Then comes the feature vectors.

	# Separting each feature by a comma and the last feature by a end of line character.
	for i in range(999):
		f.write(str(data1[0][(j,i)]) + ',')
	f.write(str(data1[0][(j,999)]) + '\n')
		
f.close()	# Closing the file