from numpy import *
from matplotlib import pyplot as plt
from skimage import io
from scipy.misc import imresize
from skimage.feature import greycomatrix

def GLCM(win, grayscale):
    coMat = zeros([grayscale,grayscale],uint32)

    # find pairs with theta = 0 and d = 1
    d = 1
    for i in range(win.shape[0]):
        for j in range(win.shape[1]-d):
            coMat[int(win[i][j])][int(win[i][j+d])] += 1

    # find variance, contrast, entropy, 
    return coMat

# zero padding
def padding(img, winSize=(31,31)):
    padded = zeros([(img.shape[0]+winSize[0]-1),
                    (img.shape[1]+winSize[1]-1)],uint8)
    padded[winSize[0]//2:winSize[0]//2+img.shape[0],winSize[1]//2:winSize[1]//2+img.shape[1]] = img

    return padded

def slidingWindow(img, grayscale=16, winSize=(31,31)):
    # padding image
    padded = padding(img, winSize)
    res = zeros(img.shape)
    #res2 = zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            win = padded[i:i+winSize[0],j:j+winSize[1]]
            #coMat = GLCM(win, grayscale)
            coMat= greycomatrix(win, [5], [0], levels=16)
            variance = var(coMat)
            #variance2 = var(coMat2)
            #print(variance)
            res[i][j] = variance
            #res2[i][j] = variance2


#    plt.subplot(1,2,1)
#    plt.imshow(res)
#    plt.subplot(1,2,2)
#    plt.imshow(res2)
#    plt.show()
    return res


# read image
z1 = io.imread('zebra_3.tif')
#z1 = z1//16

z1 = imresize(z1,(z1.shape[0],z1.shape[1]))

res = slidingWindow(z1)
tres = zeros(res.shape)
tres[res<100] = 1
plt.subplot(1,2,1)
plt.imshow(z1)
plt.subplot(1,2,2)
plt.imshow(tres)
plt.show()

