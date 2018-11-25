from numba import jit
import numpy as np

# zero padding
@jit
def padding(img, winSize=(31,31)):
    padded = np.zeros([(img.shape[0]+winSize[0]-1),
                    (img.shape[1]+winSize[1]-1)],np.uint8)
    padded[winSize[0]//2:winSize[0]//2+img.shape[0],
            winSize[1]//2:winSize[1]//2+img.shape[1]] = img
    return padded

@jit
def quantize(img, lvl=16):
    quantized = img[:]
    quantized = (quantized/np.max(img)*(lvl-1)).astype(np.int8)
    return quantized

@jit
def slidingWindow(img,d,theta,iso,featFunc,winSize=(31,31),numfeat=0):
    # padding image
    padded = padding(img, winSize)
    res = np.zeros(img.shape+tuple([numfeat]))

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
        dx = 0
        dy = 1*d
    elif theta == '45':
        dx = 1*d
        dy = 1*d
    elif theta == '90':
        dx = 1*d
        dy = 0 
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
    #w = float((win.shape[0]-dx)*(win.shape[1]-dy))
    w = 1

    # symmetrical glcm
    res = coMat/w + np.transpose(coMat)/w

    return res