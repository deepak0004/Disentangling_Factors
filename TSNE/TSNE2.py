from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import random
import pickle
import numpy as np

flatten = []
'''
with open('style_space' + '.dump', "rb") as fp: 
    flatten = pickle.load(fp)
'''
with open('Dump_Variational2/label_space_s' + '.dump', "rb") as fp: 
    flatten = pickle.load(fp)
with open('Dump_Variational2/labels' + '.dump', "rb") as fp: 
    labll = pickle.load(fp)


#labll = labll.flatten()
print len(flatten)
print labll.shape
#labll = np.argmax(labll, axis=1)
#print labll.shape
X_tsne = TSNE(learning_rate=100).fit_transform(flatten)
plt.figure(figsize=(10, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labll, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.savefig('s2_dim.png')