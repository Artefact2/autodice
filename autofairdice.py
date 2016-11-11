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
    return [ (m, n) for (m, n) in matches if m.distance < .8 * n.distance ]

def findHomographyAndInliers(kp1, kp2, matches):
    srcPoints = np.float32([ kp1[m.queryIdx].pt for (m, n) in matches ]).reshape(-1, 1, 2)
    dstPoints = np.float32([ kp2[m.trainIdx].pt for (m, n) in matches ]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 25.0)
    if mask is None:
        return [], [], 0
    
    matchesMask = [ [k,0] for k in mask.ravel().tolist()]
    return M, matchesMask, np.count_nonzero(mask)

def refmatch(img, refs):
    kp, des = keypointsAndDescriptors(img)
    
    inl = []
    for i in refs:
        matches = matched(refs[i][1], des)
        if len(matches) > 10:
            M, matchesMask, nInliers = findHomographyAndInliers(refs[i][0], kp, matches)
            inl.append((nInliers, i))

    inl = sorted(inl, reverse = True)

    if len(inl) > 1:
        if inl[0][0] > 2 * inl[1][0]:
            return inl[0][1]
        else:
            return inl[0][1] + '?'
    elif len(inl) > 0:
        return inl[0][1] + '??'
    else:
        return '?'

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

elif sys.argv[1] == "match-test":
    from matplotlib import pyplot as plt
    
    img1 = cv2.imread(sys.argv[2], -1)
    img2 = cv2.imread(sys.argv[3], -1)

    kp1, des1 = keypointsAndDescriptors(img1)
    kp2, des2 = keypointsAndDescriptors(img2)
    
    matches = matched(des1, des2)
    M, matchesMask, nInliers = findHomographyAndInliers(kp1, kp2, matches)

    print("%d matches, %d inliers" % (len(matches), nInliers))

    if nInliers > 0:
        h, w, d = img2.shape
        img4 = cv2.warpPerspective(img1, M, (w, h))
        img2 = cv2.addWeighted(img2, .7, img4, .3, 0)

    img3 = cv2.drawMatchesKnn(
        img1, kp1, img2, kp2, matches, None,
        matchesMask = matchesMask,
        singlePointColor = (0, 0, 255),
        matchColor = (0, 255, 0),
    )

    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.show()

elif sys.argv[1] == "match-ref":
    import glob
    
    refs = dict()

    for reffile in glob.glob('ref/*.png'):
        i = reffile.split('/')[1].split('.')[0]
        img = cv2.imread(reffile, -1)
        refs[i] = keypointsAndDescriptors(img)

    for i in xrange(2, len(sys.argv)):
        print "%s %s" % (sys.argv[i], refmatch(cv2.imread(sys.argv[i], -1), refs))
    
else:
    usage()
