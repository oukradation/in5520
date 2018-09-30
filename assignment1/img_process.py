import matplotlib
matplotlib.use('agg')
import sys
import numpy as np
from matplotlib import pyplot as plt

def part_texture(img, mat=(2,2)):
    tex = []
    size = img.shape[0]//2
    for i in range(mat[0]):
        for j in range(mat[1]):
            tex.append(img[i*size:i*size+size, j*size:j*size+size])
    return tex

def quantize(img, lvl=16):
    quantized = img[:]
    quantized = (quantized*(lvl-1)).astype(np.int8)
    return quantized

# zero padding
def padding(img, winSize=(31,31)):
    padded = np.zeros([(img.shape[0]+winSize[0]-1),
                    (img.shape[1]+winSize[1]-1)],np.uint8)
    padded[winSize[0]//2:winSize[0]//2+img.shape[0],
            winSize[1]//2:winSize[1]//2+img.shape[1]] = img
    return padded

def slidingWindow(img,d,theta,featFunc,winSize=(31,31)):
    # padding image
    padded = padding(img, winSize)
    res = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            win = padded[i:i+winSize[0],j:j+winSize[1]]
            coMat = GLCM(win,d,theta)
            res[i,j] = featFunc(coMat)
    return res

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

    # symmetrical glcm
    res = coMat + np.transpose(coMat)
    # normalize by number of pixel pair
    w = (win.shape[0]-dx)*(win.shape[1]-dy)
    res = res/w

    return res

def isoGLCM(win,d,grayscale=16):
    theta = ['-45','0','45','90']
    coMat = np.zeros([grayscale,grayscale],np.float)
    for t in theta:
        coMat += GLCM(win,d,t)
    return coMat/4

def IDM(coMat):
    res = 0
    for i in range(coMat.shape[0]):
        for j in range(coMat.shape[1]):
            res += 1/(1+(i-j)**2) * coMat[i,j]
    return res

def inertia(coMat):
    res = 0
    for i in range(coMat.shape[0]):
        for j in range(coMat.shape[1]):
            res += (i-j)**2 * coMat[i,j]
    return res

def shade(coMat):
    res = 0

    # ux and uy same for symmetrical GLCM
    u = ux(coMat)
    
    for i in range(coMat.shape[0]):
        for j in range(coMat.shape[1]):
            res += (i + j - 2*u)**3 * coMat[i,j] 
    return res
            
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



img = plt.imread('mosaic1.png')
q = quantize(img)

d = int(sys.argv[1])
theta = sys.argv[2]
win = int(sys.argv[3])
func = sys.argv[4]

print('Saving to %s_d_%d_th_%s_win_%d.png'%(func,d,theta,win))

featureMap = slidingWindow(q,d,theta,eval(func),(win,win))
plt.imshow(featureMap)
plt.savefig('../%s_d_%d_th_%s_win_%d.png'%(func,d,theta,win))


#tex = part_texture(img)
#
#d = 1
#theta = '-45'
#
#win = 31
#
#for i in range(4):
#    q = quantize(tex[i])
#    g = GLCM(q[:win,:win],d,theta)
#
#    assert g.all()==np.transpose(g).all(), 'GLCM not symmetrical'
#
#    print(i,'IDM : %.02f \t Inertia : %.02f, \t Shade : %.02f'%(IDM(g),inertia(g),shade(g)))
#    
#    plt.subplot(2,4,i+1)
#    plt.imshow(g)
#    plt.subplot(2,4,i+5)
#    plt.imshow(tex[i][:win,:win])
#
#plt.savefig('../d_%d_th_%s_win_%d.png'%(d,theta,win))
