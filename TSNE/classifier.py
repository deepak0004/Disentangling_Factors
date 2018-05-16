from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import random
import pickle
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVR

flatten = []

with open('Dump_Variational2/label_space_z' + '.dump', "rb") as fp: 
    flatten = pickle.load(fp)
with open('Dump_Variational2/labels' + '.dump', "rb") as fp: 
    labll = pickle.load(fp)

train_x = []
train_y = []
leng = 64
pp = 0
i = 0
trainingsp = ( len(flatten)*0.8 )
#print 'len(flatten): ', len(flatten)
#print 'flatten[0]: ', flatten[0]
while( i < trainingsp ):
    oioi = []
    oioi.append( flatten[i] )
    train_x.append( oioi )
    train_y.append( labll[pp] )
    i += 1
    pp += 1 

trainvector = np.reshape( train_x, (len(train_x), leng) )
trainlabel = np.reshape( train_y, (len(train_y), 1) )

print('Defining')
#clf2 = svm.LinearSVC()  
clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(256,10), random_state=1)
print('Training')
clf2.fit(trainvector, trainlabel.ravel())

t1 = []
t2 = []
correct = 0
rejected = 0
noofval = 0
while( i < len(flatten) ):
    t1.append( labll[pp] )
    noofval += 1
    img = flatten[i]
    try:
        lab = clf2.predict([img])
        t2.append( lab )
        if( lab==labll[pp] ):
          correct += 1
          
    except Exception: 
        print img
        rejected += 1
        pass

    i += 1
    pp += 1  

#print t2
#print rejected

print 'Accuracy: ', (float(correct)/noofval)
print 'F1 score: ', f1_score(t1, t2, average='macro')