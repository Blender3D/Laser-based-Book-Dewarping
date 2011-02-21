'''
import sys, cv

image = cv.LoadImage(sys.argv[1])
dewarped = cv.CreateImage(cv.GetSize(image), 8, 3)

found, corners = cv.FindChessboardCorners(image, (5, 5))

cv.DrawChessboardCorners(dewarped, (5, 5), corners, found)

cv.SaveImage(sys.argv[2], dewarped
'''

import cv # opencv 2.2.0(?)
import glob

# Location of chessboard images, file to correct and corrected file.
chessboard_path = './open*'
input_file = 'IMG_0885.JPG'
output_file = 'IMG_0885.dist.JPG'

# Number of intersections. (squares-1).
# Change this is you change chessboard image.
num_x_ints = 5
num_y_ints = 8

num_pts = num_x_ints * num_y_ints

def get_corners(mono, refine = False):
    (ok, corners) = cv.FindChessboardCorners(mono, (num_x_ints, num_y_ints), cv.CV_CALIB_CB_ADAPTIVE_THRESH | cv.CV_CALIB_CB_NORMALIZE_IMAGE)
    if refine and ok:
        corners = cv.FindCornerSubPix(mono, corners, (5,5), (-1,-1), ( cv.CV_TERMCRIT_EPS+cv.CV_TERMCRIT_ITER, 30, 0.1 ))
    return (ok, corners)

def mk_object_points(nimages, squaresize = 1):
    opts = cv.CreateMat(nimages * num_pts, 3, cv.CV_32FC1)
    for i in range(nimages):
        for j in range(num_pts):
            opts[i * num_pts + j, 0] = (j / num_x_ints) * squaresize
            opts[i * num_pts + j, 1] = (j % num_x_ints) * squaresize
            opts[i * num_pts + j, 2] = 0
    return opts

def mk_image_points(goodcorners):
    ipts = cv.CreateMat(len(goodcorners) * num_pts, 2, cv.CV_32FC1)
    for (i, co) in enumerate(goodcorners):
        for j in range(num_pts):
            ipts[i * num_pts + j, 0] = co[j][0]
            ipts[i * num_pts + j, 1] = co[j][1]
    return ipts

def mk_point_counts(nimages):
    npts = cv.CreateMat(nimages, 1, cv.CV_32SC1)
    for i in range(nimages):
        npts[i, 0] = num_pts
    return npts


files = glob.glob(chessboard_path)
images = [cv.LoadImage(i, cv.CV_LOAD_IMAGE_COLOR) for i in files]
size = cv.GetSize(images[0])

corners = [get_corners(i) for i in images]
goodcorners = [co for (im, (ok, co)) in zip(images, corners) if ok]

ipts = mk_image_points(goodcorners)
opts = mk_object_points(len(goodcorners), .1)
npts = mk_point_counts(len(goodcorners))

intrinsics = cv.CreateMat(3, 3, cv.CV_64FC1)
distortion = cv.CreateMat(4, 1, cv.CV_64FC1)
cv.SetZero(intrinsics)
cv.SetZero(distortion)

# focal lengths have 1/1 ratio
intrinsics[0,0] = 1.0
intrinsics[1,1] = 1.0

cv.CalibrateCamera2(opts, ipts, npts,
           cv.GetSize(images[0]),
           intrinsics,
           distortion,
           cv.CreateMat(len(goodcorners), 3, cv.CV_32FC1),
           cv.CreateMat(len(goodcorners), 3, cv.CV_32FC1),
           flags = cv.CV_CALIB_ZERO_TANGENT_DIST) # cv.CV_CALIB_ZERO_TANGENT_DIST)

mapx = cv.CreateImage((size[0], size[1]), cv.IPL_DEPTH_32F, 1)
mapy = cv.CreateImage((size[0], size[1]), cv.IPL_DEPTH_32F, 1)
cv.InitUndistortMap(intrinsics, distortion, mapx, mapy)

# ---------------------------------------------------------------------

# Loop this to correct multiple images.
img = cv.LoadImage(input_file)
r = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U, 3);
cv.Remap(img, r, mapx, mapy)
cv.SaveImage(output_file, r);
