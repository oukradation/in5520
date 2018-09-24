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

def slidingWindow(img, grayscale=16, winSize=(31,31)):
    # padding image
    padded = padding(img, winSize)
    res = np.zeros(img.shape)
    #res2 = zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            win = padded[i:i+winSize[0],j:j+winSize[1]]

    return res

def GLCM(win, grayscale=16):
    coMat = np.zeros([grayscale,grayscale],np.uint16)

    d = 1
    # find pairs with theta = 0 and d = 1
    for i in range(win.shape[0]):
        for j in range(win.shape[1]-d):
            coMat[win[i][j]][win[i][j+d]] += 1
    return coMat



img = plt.imread('mosaic1.png')
tex = part_texture(img)

for i in range(4):
    q = quantize(tex[i])
    g = GLCM(q[:31,:31])
    plt.subplot(2,4,i+1)
    plt.imshow(g)
    plt.subplot(2,4,i+5)
    plt.imshow(tex[i][:31,:31])
plt.show()
