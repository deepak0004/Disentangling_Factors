from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import random
import pickle
import numpy as np

def normalize(v):
     norm=np.linalg.norm(v, ord=1)
     if norm==0:
         norm=np.finfo(v.dtype).eps
     return v/norm

flatten = []
flattensec = []
'''
with open('style_space' + '.dump', "rb") as fp: 
    flatten = pickle.load(fp)
'''
with open('Dump_Variational1/label_space_s' + '.dump', "rb") as fp: 
    flatten = pickle.load(fp)
with open('Dump_Variational2/label_space_s' + '.dump', "rb") as fp: 
    flattensec = pickle.load(fp)
with open('Dump_Variational1/labels' + '.dump', "rb") as fp: 
    labll = pickle.load(fp)
with open('Dump_Variational2/labels' + '.dump', "rb") as fp: 
    labllsec = pickle.load(fp)

#print flatten[0]
'''
for i in range(len(flatten)):
    su = 0
    if( labll[i]!=1 or labll[i]!=2 ):
       for j in range(len(flatten[i])):
           flatten[i][j] = 0
       continue
    coun += 1
    for j in range(len(flatten[i])):
        su += flatten[i][j]
    #print su
    for j in range(len(flatten[i])):
        flatten[i][j] /= su
'''
labell2 = []
flatten2 = []
coun2 = 0
coun3 = 0
coun4 = 0
coun5 = 0
for i in range(len(flatten)):
    if( labllsec[i]==1 and coun2<100 ):
       labell2.append(0)
       flatten2.extend( flatten[i] )
       coun2 += 1
    elif( labllsec[i]==2 and coun3<100 ):
       labell2.append(0)
       flatten2.extend( flatten[i] )
       coun3 += 1
    if( labll[i]==1 and coun4<100 ):
       labell2.append(1)
       flatten2.extend( flattensec[i] )
       coun4 += 1
    elif( labll[i]==2 and coun5<100 ):
       labell2.append(1)
       flatten2.extend( flattensec[i] )
       coun5 += 1

print coun2, " ", coun3, " ", coun4, " ", coun5 
#print coun
#print flatten2[0]
#labll = labll.flatten()
#print len(flatten2)
#print labll.shape
#labll = np.argmax(labll, axis=1)
#print labll.shape
flatten = flatten2
labll = labell2

flatten = np.reshape(flatten, [400, 64])

#print labll
#print flatten[199]

X_tsne = TSNE(learning_rate=100).fit_transform(flatten)

su = 0
for i in range(len(X_tsne)):
	su += X_tsne[i][0]
su /= len(X_tsne)
print su
for i in range(len(X_tsne)):
	X_tsne[i][0] -= su
su = 0
for i in range(len(X_tsne)):
	su += X_tsne[i][1]
su /= len(X_tsne)
print su
for i in range(len(X_tsne)):
	X_tsne[i][1] -= su

X_tsne[:, 0] = normalize(X_tsne[:, 0])    
X_tsne[:, 1] = normalize(X_tsne[:, 1])    

plt.figure(figsize=(10, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labll, cmap=plt.cm.get_cmap("jet", 2))
plt.colorbar(ticks=range(2))
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.rcParams["font.family"] = "Times New Roman"
plt.xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
plt.yticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])

plt.clim(-0.5, 9.5)
plt.savefig('s1_dim.png')