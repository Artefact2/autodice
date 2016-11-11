#!/usr/bin/env python2

# Author: Romain "Artefact2" Dalmaso <artefact2@gmail.com>

# This program is free software. It comes without any warranty, to the
# extent permitted by applicable law. You can redistribute it and/or
# modify it under the terms of the Do What The Fuck You Want To Public
# License, Version 2, as published by Sam Hocevar. See
# http://sam.zoy.org/wtfpl/COPYING for more details.

import sys
import math
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

    M, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 10.0)
    if mask is None:
        return [], [], 0
    
    matchesMask = [ [k,0] for k in mask.ravel().tolist()]
    return M, matchesMask, np.count_nonzero(mask)

def distanceToCenter(x, y, w, h):
    dx = x - .5 * (w-1)
    dy = y - .5 * (h-1)
    return math.sqrt(dx*dx + dy*dy)

def scoreMatches(matches, mask, kp1, w1, h1, kp2, w2, h2):
    score = 0.0
    
    for i, (m, n) in enumerate(matches):
        if not mask[i][0]:
            continue
        
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        r1 = distanceToCenter(pt1[0], pt1[1], w1, h1)
        r2 = distanceToCenter(pt2[0], pt2[1], w2, h2)

        score += 1.0 / (1.0 + max(0, abs(r1 / 30.0) - 1)) / (1.0 + max(0, abs((r2-r1) / 10.0) - 1))
        
    return score

def scoreCenterDiff(img1, img2, p):
    h, w, d = img1.shape
    y1 = int((.5 - p / 2.0) * h)
    y2 = int((.5 + p / 2.0) * h)
    x1 = int((.5 - p / 2.0) * w)
    x2 = int((.5 + p / 2.0) * w)

    s1 = img1[y1:y2,x1:x2].view(dtype=np.int8)
    s2 = img2[y1:y2,x1:x2].view(dtype=np.int8)
    
    return np.linalg.norm(s2 - s1) / ((y2 - y1)*(x2 - x1))

def refmatch(img, refdata):
    kp, des = keypointsAndDescriptors(img)
    
    inl = []
    for i in refdata:
        matches = matched(refdata[i][0][1], des)
        if len(matches) <= 10:
            continue
        
        M, matchesMask, nInliers = findHomographyAndInliers(refdata[i][0][0], kp, matches)

        if nInliers <= 10:
            continue

        h1, w1, d = refdata[i][1].shape
        h2, w2, d = img.shape

        warped = cv2.warpPerspective(refdata[i][1], M, (w2, h2))
        
        inl.append((
            scoreMatches(matches, matchesMask, refdata[i][0][0], w1, h1, kp, w2, h2) - 10.0 * scoreCenterDiff(img, warped, .333),
            i,
        ))

    inl = sorted(inl, reverse = True)

    if len(inl) > 1:
        if inl[0][0] > 1.5 * inl[1][0]:
            return inl[0][1]
        else:
            print(inl)
            return inl[0][1] + '?'
            
    elif len(inl) > 0:
        return inl[0][1]
    else:
        return '???'

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
    plt.imshow(cv2.Canny(img, 100, 200), cmap='gray')
    plt.show()

elif sys.argv[1] == "match-test":
    from matplotlib import pyplot as plt
    
    img1 = cv2.imread(sys.argv[2], -1)
    img2 = cv2.imread(sys.argv[3], -1)

    h1, w1, d = img1.shape
    h2, w2, d = img2.shape

    kp1, des1 = keypointsAndDescriptors(img1)
    kp2, des2 = keypointsAndDescriptors(img2)
    
    matches = matched(des1, des2)
    M, matchesMask, nInliers = findHomographyAndInliers(kp1, kp2, matches)

    if nInliers > 0:
        img4 = cv2.warpPerspective(img1, M, (w2, h2))
        img5 = cv2.addWeighted(img2, .7, img4, .3, 0)
        cdScore = scoreCenterDiff(img4, img2, .333)
    else:
        cdScore = 0.0

    print("%d matches, %d inliers, final scores: (%f, %f)" % (
        len(matches),
        nInliers,
        scoreMatches(matches, matchesMask, kp1, w1, h2, kp2, w2, h2),
        cdScore,
    ))

    img3 = cv2.drawMatchesKnn(
        img1, kp1, img5, kp2, matches, None,
        matchesMask = matchesMask,
        singlePointColor = (0, 0, 255),
        matchColor = (0, 255, 0),
    )

    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.show()

elif sys.argv[1] == "match-ref":
    import glob
    
    refdata = dict()

    for reffile in glob.glob('ref/*.png'):
        i = reffile.split('/')[1].split('.')[0]
        img = cv2.imread(reffile, -1)
        refdata[i] = (
            keypointsAndDescriptors(img),
            img,
        )

    for i in xrange(2, len(sys.argv)):
        print "%s %s" % (sys.argv[i], refmatch(cv2.imread(sys.argv[i], -1), refdata))
    
else:
    usage()
