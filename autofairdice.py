#!/usr/bin/env python2

import sys
import numpy as np
import cv2

sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers = 1)
bf = cv2.BFMatcher()

def usage():
    print("Usage:")
    print("%s autocrop <in.png> <out.png>" % sys.argv[0])
    print("%s autocrop-test <in.png>" % sys.argv[0])
    print("%s preprocess-ref" % sys.argv[0])
    print("%s match-test <A.png> <B.png>" % sys.argv[0])
    print("%s match-ref <image.png> ..." % sys.argv[0])
    sys.exit()

def autocrop(img, t1 = 100, t2 = 200):
    edges = cv2.Canny(img, t1, t2)
    x, y, w, h = cv2.boundingRect(edges)
    return img[y:y+h,x:x+w]

def keypointsAndDescriptors(img):
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def matched(des1, des2):
    matches = bf.knnMatch(des1, des2, k = 2)
    return [ (m, n) for (m, n) in matches if m.distance < .7 * n.distance ]

def refmatch(img, ref):
    kp, des = keypointsAndDescriptors(img)
    
    goodmatches = []
    for i in refs.files:
        matches = matched(des, refs[i])
        goodmatches.append((len(matches), i))

    goodmatches = sorted(goodmatches, reverse = True)
    
    if goodmatches[0][0] > 1.5 * goodmatches[1][0]:
        return goodmatches[0][1]
    else:
        return goodmatches[0][1] + '?'

if len(sys.argv) == 1:
    usage()

if sys.argv[1] == "autocrop":    
    img = cv2.imread(sys.argv[2], -1)
    cv2.imwrite(sys.argv[3], autocrop(img))
    
elif sys.argv[1] == "autocrop-test":
    from matplotlib import pyplot as plt
    
    img = cv2.imread(sys.argv[2], -1)
    plt.imshow(cv2.cvtColor(autocrop(img), cv2.COLOR_BGR2RGB))
    plt.show()

elif sys.argv[1] == "preprocess-ref":
    import glob

    descriptors = dict()

    for reffile in glob.iglob('ref/*.png'):
        img = cv2.imread(reffile, 0)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        descriptors[reffile.split('/')[1].split('.')[0]] = des

    np.savez('ref-descriptors.npz', **descriptors)

elif sys.argv[1] == "match-test":
    from matplotlib import pyplot as plt
    
    img1 = cv2.imread(sys.argv[2], 0)
    img2 = cv2.imread(sys.argv[3], 0)

    kp1, des1 = keypointsAndDescriptors(img1)
    kp2, des2 = keypointsAndDescriptors(img2)
    matches = matched(des1, des2)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None)

    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.show()

elif sys.argv[1] == "match-ref":
    refs = np.load('ref-descriptors.npz')

    for i in xrange(2, len(sys.argv)):
        print "%s %s" % (sys.argv[i], refmatch(cv2.imread(sys.argv[i], 0), refs))
    
else:
    usage()
