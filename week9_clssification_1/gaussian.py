#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import collections
import sys

class gaussianClassifier():
    def __init__(self, inputs,  numVar = 1):
        self.numVar = numVar

        self.u = np.mean(inputs,axis=1)
        self.cov = np.cov(inputs) 
        self.var = np.var(inputs)
    
    def classify(self, x, Pw):
        if self.numVar > 1 :
            #return ( 1 / np.sqrt( (2*np.pi)**self.numVar * np.linalg.det( self.cov ) ) 
            #        * np.exp( -0.5 * np.matmul(np.matmul(( x - self.u ).transpose(),np.linalg.inv( self.cov )), ( x - self.u ) ) ) ) 
            # log
            return ( -0.5 * np.matmul(np.matmul(( x - self.u ).transpose(),np.linalg.inv( self.cov )), ( x - self.u ) )
                    - self.numVar/2 * np.log(2*np.pi)  -0.5 * np.log(np.linalg.det(self.cov)) )

        elif self.numVar == 1:
            return 1 / np.sqrt( 2*np.pi*self.var ) * np.exp(-.5 * (x - self.u)**2 / self.var ) * Pw

class classifier():
    def __init__(self, numClass):
        self.numClass = numClass
        self.gaussian = []

    # trainset MxNxL where M corresponds to each classes, and N number of variable, L length of feature vector
    def train(self, trainset):
        # for each class
        for i in range(self.numClass):
            # instanciate gaussianCLF
            self.gaussian.append(gaussianClassifier(trainset[i], trainset[i].shape[0]))


    def classify(self, feature):
        prb = []
        for i in range(self.numClass):
            prb.append(self.gaussian[i].classify(feature, 1))
        return prb.index(max(prb))

    def validate(self, validset, label):
        pass


if __name__ == '__main__':
    # open train mask
    train_mask = plt.imread('tm_train.png')

    # convert mask image into 4 discrete class [1,4]
    train_mask = (train_mask/np.max(train_mask)*4).astype(np.uint8)

    # do same for test mask
    test_mask = plt.imread('tm_test.png')
    test_mask = (test_mask/np.max(test_mask)*4).astype(np.uint8)
    

    imgs = []
    for i in range(6):
        imgs.append(plt.imread('tm%d.png'%(i+1))*100)
    imgs = np.array(imgs)

    #make feature set for training
    feats = []
    for i in range(4):
        feats.append([])
        for j in range(6):
            #feats[i].append((imgs[j][np.where(train_mask==i+1)]).flatten())
            feats[i].append((imgs[j]*(train_mask==i+1)).flatten())
        feats[i] = np.array(feats[i])
    #feats = np.array(feats)

    #train classfier
    clf = classifier(4)
    clf.train(feats)

    #validate classifer on test set
    for cla in range(1,5):
        acc = 0
        tot = 0
        for i in range(imgs[0].shape[0]):
            for j in range(imgs[0].shape[1]):
                if test_mask[i,j] == cla:
                    res = clf.classify(imgs[:,i,j])+1
                    tot += 1
                    if res == cla:
                        acc += 1

        print(cla, 'acc', acc/tot)

    sys.exit()

    # classify the whole image
    res = np.zeros(imgs[0].shape)
    for i in range(imgs[0].shape[0]):
        for j in range(imgs[0].shape[1]):
            res[i,j] = clf.classify(imgs[:,i,j])

    plt.subplot(121)
    plt.imshow(res)
    plt.subplot(122)
    plt.imshow(imgs[4])
    plt.show()


#    # MxNx2 where M is imgNr, N is class, and tuple of (mean, var)
#    dist = []
#    for i in range(6):
#        dist.append([])
#        for j in range(4):
#            tmp = imgs[i]*(train_mask==j+1)
#            dist[i].append(gaussianClassifier(tmp[np.where(tmp!=0)]))
#
#    res = []
#    for im in range(6):
#        res.append([])
#        for mask in range(1,5):
#            t = imgs[im]*(test_mask==mask)
#            #t = imgs[im]
#            temp = np.zeros(t.shape)
#            for i in range(len(t)):
#                for j in range(len(t[0])):
#                    if t[i][j] != 0:
#                        tmp = []
#                        for k in range(4):
#                            tmp.append(dist[im][k].classify(t[i][j],1))
#                        temp[i,j] = tmp.index(max(tmp))+1
#
#
#            flat = temp[np.where(temp!=0)].flatten()
#            dominant = collections.Counter(flat)
#
#            if dominant.get(mask):
#                acc = dominant.get(mask)/len(flat)
#            else:
#                acc = 0
#
#            crr = False
#            if dominant.most_common()[0][0] == mask:
#                crr = True
#
#            print(im+1, mask, acc, crr)
#            res[im].append((crr,acc))
#            #plt.imshow(temp)
#            #plt.title('img: %d mask: %d est.class : %d'%(im+1, mask, dominant.most_common()[0][0]))
#            #plt.savefig('img_%d_mask_%d_estClass_%d.png'%(im+1, mask, dominant.most_common()[0][0]))
#
#    for i in range(len(res)):
#        print(res[i])
#



