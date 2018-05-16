from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import random
import pickle

flatten = []

'''
with open('style_space' + '.dump', "rb") as fp: 
    flatten = pickle.load(fp)
'''
with open('Dump_Variational1/label_space_z' + '.dump', "rb") as fp: 
    flatten = pickle.load(fp)
with open('Dump_Variational1/labels' + '.dump', "rb") as fp: 
    labll = pickle.load(fp)

flatten.flatten()
labll = labll.flatten()
X_tsne = TSNE(learning_rate=100).fit_transform(flatten)
plt.figure(figsize=(10, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labll, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.savefig('z1dim.png')