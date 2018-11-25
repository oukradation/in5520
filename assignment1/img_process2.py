import matplotlib
matplotlib.use('agg')
import sys
import numpy as np
from matplotlib import pyplot as plt
from numba import jit

@jit
def part_texture(img, mat=(2,2)):
    tex = []
    size = img.shape[0]//2
    for i in range(mat[0]):
        for j in range(mat[1]):
            tex.append(img[i*size:i*size+size, j*size:j*size+size])
    return tex

@jit
def quantize(img, lvl=16):
    quantized = img[:]
    quantized = (quantized*(lvl-1)).astype(np.int8)
    return quantized

# zero padding
@jit
def padding(img, winSize=(31,31)):
    padded = np.zeros([(img.shape[0]+winSize[0]-1),
                    (img.shape[1]+winSize[1]-1)],np.uint8)
    padded[winSize[0]//2:winSize[0]//2+img.shape[0],
            winSize[1]//2:winSize[1]//2+img.shape[1]] = img
    return padded

@jit
def slidingWindow(img,d,theta,iso,featFunc,winSize=(31,31)):
    # padding image
    padded = padding(img, winSize)
    res = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            win = padded[i:i+winSize[0],j:j+winSize[1]]
            if iso:
                coMat = isoGLCM(win,d)
            else:
                coMat = GLCM(win,d,theta)
            res[i,j] = featFunc(coMat)
    return res

@jit
def GLCM(win, d, theta='0', grayscale=16):
    assert theta in ['-45','0','45','90'], 'unsupported theta'
    coMat = np.zeros([grayscale,grayscale],np.uint16)

    if theta == '0':
        dx = 1*d
        dy = 0 
    elif theta == '45':
        dx = 1*d
        dy = 1*d
    elif theta == '90':
        dx = 0
        dy = 1*d
    else:
        dx = 1*d
        dy = 1*d
        # mirror
        win = np.flip(win[:],axis=1)

    # find pairs with theta = 0 and d = 1
    for i in range(win.shape[0]-dy):
        for j in range(win.shape[1]-dx):
            coMat[win[i][j]][win[i+dy][j+dx]] += 1

    # normalize by number of pixel pair
    w = float((win.shape[0]-dx)*(win.shape[1]-dy))

    # symmetrical glcm
    res = coMat/w + np.transpose(coMat)/w

    return res
@jit
def isoGLCM(win,d,grayscale=16):
    theta = ['-45','0','45','90']
    coMat = np.zeros([grayscale,grayscale],np.float)
    for t in theta:
        coMat += GLCM(win,d,t)
    return coMat/4
@jit
def IDM(coMat):
    res = 0
    for i in range(coMat.shape[0]):
        for j in range(coMat.shape[1]):
            res += 1/(1+(i-j)**2) * coMat[i,j]
    return res
@jit
def inertia(coMat):
    res = 0
    for i in range(coMat.shape[0]):
        for j in range(coMat.shape[1]):
            res += (i-j)**2 * coMat[i,j]
    return res
@jit
def shade(coMat):
    res = 0

    # ux and uy same for symmetrical GLCM
    u = ux(coMat)
    
    for i in range(coMat.shape[0]):
        for j in range(coMat.shape[1]):
            res += (i + j - 2*u)**3 * coMat[i,j] 
    return res
@jit            
def ux(img):
    res = 0
    for i in range(img.shape[0]):
        res += i*np.sum(img[i])
    return res

def uy(img):
    img = np.transpose(img[:])
    res = 0
    for i in range(img.shape[0]):
        res += i*np.sum(img[i])
    return res

# Visualize GLCM for a texture patch size of given window size
def visualizeGLCM(filename, d,theta,win,iso=False):
    img = plt.imread(filename)
    q = quantize(img)
    tex = part_texture(img)
    
    for i in range(4):
        q = quantize(tex[i])
        if iso:
            g = GLCM(q[:win,:win],d,theta)
        else:
            g = isoGLCM(q[:win,:win],d)
    
    
        print(i,'IDM : %.02f \t Inertia : %.02f, \t Shade : %.02f'%(IDM(g),inertia(g),shade(g)))
        plt.subplot(1,2,1)
        plt.imshow(g,interpolation='none',cmap=plt.get_cmap('Spectral'))
        plt.title('d:%d theta:%s win:%d'%(d,theta,win))
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(tex[i][:win,:win],cmap=plt.get_cmap('gray'))
        plt.savefig('../%s%d_d_%d_th_%s_win_%d.png'%(filename,i+1,d,theta,win))
        plt.clf()

# take a numpy file which contains feature map 
#   apply global threshold of given range <lower,upper>
#   create mask and outputs image to a file
def globalThreshold(featureFile, lower, upper, mosaic, fromFile=False):
    if fromFile:
        featureMap = np.load(featureFile)
    else:
        featureMap = featureFile

    img = plt.imread('mosaic%s.png'%mosaic)
    mask = (featureMap < float(lower))*(featureMap > float(upper))
    plt.subplot(131)
    plt.imshow(featureMap, cmap=plt.get_cmap('Spectral'))
    plt.subplot(132)
    plt.imshow(mask,cmap=plt.get_cmap('gray'))
    title = 'Threshold : %.02f < T < %.02f'%(float(lower),float(upper))
    plt.title(title)
    plt.subplot(133)
    plt.imshow(img*mask,cmap=plt.get_cmap('gray'))
    plt.show()
    plt.savefig('%s%s.png'%(mosaic+featureFile[:-4]+lower+upper,title))
    
# create feature map for given parameter
# outputs feature map image and numpy file containing the feature map
def createFeatureMap(imgNum, d, theta, win, func, iso):
    filename = 'p%s_%s_d_%d_th_%s_win_%d_iso_%d'%(imgNum,func,d,theta,win,iso)
    
    print'Saving to ', filename
    
    img = plt.imread('mosaic%s.png'%imgNum)
    q = quantize(img)
    
    featureMap = slidingWindow(q,d,theta,iso,eval(func),(win,win))
    plt.imshow(featureMap, cmap=plt.get_cmap('Spectral'))
    plt.colorbar()
    plt.savefig('%s.png'%filename)
    np.save('%s.npy'%filename, featureMap)
    
    return featureMap

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print 'Argument\n\t img d theta win featFunc iso'
        sys.exit()
    
    imgNum = sys.argv[1]
    d = int(sys.argv[2])
    theta = sys.argv[3]
    win = int(sys.argv[4])
    func = sys.argv[5]
    iso = int(sys.argv[6])

    featureMap = createFeatureMap(imgNum, d, theta, win, func, iso)
    globalThreshold(featureMap, 0, 1.5, imgNum)

        
