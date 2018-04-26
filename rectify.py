import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
img1 = cv2.imread('/home/xu/data/view_synthesis/pvhm_test/rendering_result/A/00000.png',0)
img2 = cv2.imread('/home/xu/data/view_synthesis/pvhm_test/rendering_result/B/00000.png',0)


cameraMatrix1 =  np.array(
            [128. / 32. * 60, 0., 64., \
             0., 128. / 32. * 60, 64., \
             0., 0., 1.],dtype=np.float32).reshape((3,3))

cameraMatrix2 =  np.array(
            [128. / 32. * 60, 0., 64., \
             0., 128. / 32. * 60, 64., \
             0., 0., 1.],dtype=np.float32).reshape((3,3))

# intrinsics ?
distCoeffs1 = np.zeros((1,4),dtype=np.float32)
distCoeffs2 = np.zeros((1,4),dtype=np.float32)
imageSize = (128,128)

def R(theta):
    return np.array(
            [np.cos(theta), 0, np.sin(theta), 0, 1, 0, -np.sin(theta), 0, np.cos(theta)]).reshape((3,3))

def T(theta):
    return 4 * np.array([-np.sin(theta),0,1-np.cos(theta)])
extrinsics1 = np.eye(3,4)
extrinsics1 = np.eye(3,4)
angle = np.pi/4.
R1,R2,M1,M2,_,_,_ = cv2.stereoRectify(cameraMatrix1,  distCoeffs1,cameraMatrix2, distCoeffs2, imageSize,
                 R(angle), T(angle), None, 1.0, None, None)

map11,map12 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1.astype(np.float32), M1.astype(np.float32), imageSize, cv2.CV_32FC1)
map21,map22 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2.astype(np.float32), M2.astype(np.float32), imageSize, cv2.CV_32FC1)
print type(map11[0,0])
img1_rect = cv2.remap(img1,map11.astype(np.float32)+100,map12.astype(np.float32),cv2.INTER_NEAREST)
img2_rect = cv2.remap(img2,map21.astype(np.float32)-100,map22.astype(np.float32),cv2.INTER_NEAREST)
print img2_rect

img1 = img1_rect
img2 = img2_rect

sift = cv2.ORB_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.8*n.distance:
        good.append([m])
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2,outImg=None)
plt.imshow(img3),plt.show(),\
plt.colorbar()

