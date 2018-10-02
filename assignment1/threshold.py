import matplotlib
matplotlib.use('agg')
import numpy as np
from matplotlib import pyplot as plt
import sys

featureMap = np.load(sys.argv[1])
img = plt.imread('mosaic%s.png'%sys.argv[4])
mask = (featureMap < float(sys.argv[3]))*(featureMap > float(sys.argv[2]))
plt.subplot(131)
plt.imshow(featureMap, cmap=plt.get_cmap('Spectral'))
plt.subplot(132)
plt.imshow(mask,cmap=plt.get_cmap('gray'))
title = 'Threshold : %.02f < T < %.02f'%(float(sys.argv[2]),float(sys.argv[3]))
plt.title(title)
plt.subplot(133)
plt.imshow(img*mask,cmap=plt.get_cmap('gray'))
plt.show()
plt.savefig('%s%s.png'%(sys.argv[1][3:-4],title))
