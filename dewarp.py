import os, sys
from math import *

import cv
import Image, ImageOps, ImageDraw
import Gnuplot
import numpy

#Numpy is a Python numerical library (matrices, etc.), so this just smooths data. This is basically all that I use numpy for.

def numpy_smooth(x, window_len = 11, window = 'hanning'):
  if window_len < 3:  return x
  
  if window == 'flat':
    w = numpy.ones(window_len, 'd')
  else:
    w = eval('numpy.' + window + '(window_len)')
  
  return numpy.convolve(w / w.sum(), numpy.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]], mode = 'same')[window_len: -window_len + 1]



imagePath = sys.argv[1]

image = cv.LoadImage(imagePath)
cv_image = image # I use this as a seperate canvas to draw on.

imageThreshold = cv.CreateImage(cv.GetSize(image), 8, 1) # The last number represents a binary image. It's the channel-depth.

cv.NamedWindow('Test', 0)
cv.ResizeWindow('Test', 1000, 600)

channelG = cv.CreateImage(cv.GetSize(image), 8, 1) # Initializes blank image holders for the channel separation.
channelB = cv.CreateImage(cv.GetSize(image), 8, 1)

cv.Split(image, cv.CreateImage(cv.GetSize(image), 8, 1), channelG, channelB, None) # Splits the input image into channels (RGB). I only use B and G because the laser is green, not red.

cv.Sub(channelG, channelB, imageThreshold) # Subtracts the channels. Since the green laser's pixels are basically have a G value of 255, they stand out. Uncomment the next two lines to see what I mean.
#cv.ShowImage('Test', imageThreshold)
#cv.WaitKey(0)

cv.InRangeS(imageThreshold, cv.Scalar(24, 24, 24), cv.Scalar(255, 255, 255), imageThreshold) # The channel seperation doesn't make things bitonal. This is something I need to fix, as it depends on a fixed value, but this just thresholds everything between those RGB values into white.

cv.ShowImage('Test', imageThreshold)
cv.WaitKey(0)

cv.Smooth(imageThreshold, imageThreshold, smoothtype = cv.CV_MEDIAN, param1 = 9) # This median smoothing filter is really handy. It averages the values of each pixel within the window, so any noise gets truned into black (the image is bitonal, so if it's not white, it's black).

cv.ShowImage('Test', imageThreshold)
cv.WaitKey(0)

image = Image.fromstring('L', cv.GetSize(imageThreshold), imageThreshold.tostring()) # I have to get rid of this dependency to Python's imaging library, but this just re-inits the thresholded image.
pixels = image.load() # Loads the pixel data, as this object is iterable.


# Extraction of data points happens below

top = [] # I use two data sets for the two lines.
bottom = []

for x in xrange(image.size[0]): # Loops over pixel columns
  intersects = [] # Resets the intersection list. This is what I mainly use for line-detection.
  
  for y in reversed(xrange(image.size[1] / 2, image.size[1])): # This loops backwards halfway through the list to find the top curve's shape
    if pixels[x, y] == 255:
      intersects.append(y)
    
  if len(intersects) > 1: # Just for safety
    top.append((intersects[-1] + intersects[0]) / 2)
  elif len(top) > 0:
    top.append(top[-1]) # If the laser line misses a data point, fill it in with the previous one.
  else:
    top.append(3 * image.size[1] / 4) # If no data points exist at all, 3/4 of image height is the first value.

for x in xrange(image.size[0]): # A second loop. These two loops interfered when I combined them, so I have to do it twice.
  intersects = []
  
  for y in xrange(0, image.size[1] / 2):
    if pixels[x, y] == 255:
      intersects.append(y)
  
  if len(intersects) > 1:
    bottom.append((intersects[-1] + intersects[0]) / 2)
  elif len(bottom) > 0:
    bottom.append(bottom[-1])
  else:
    bottom.append(image.size[1] / 4)

# No smoothing for now...
#top = numpy_smooth(numpy.array(top), window_len = 51)
#bottom = numpy_smooth(numpy.array(top), window_len = 51)


verticies = []

for i in xrange(0, len(top), 2): # Constructs the 3D model. The shape is the only thing that matters, so these numbers work for now.
  verticies.append([(len(top) / 2 - float(i)) / 1000.0, 0.5, float(top[i] - min(top)) / 1000.0])
  verticies.append([(len(top) / 2 - float(i)) / 1000.0, -0.5, float(max(bottom) - bottom[i]) / 1000.0])


class Point: # Just a simple Point or Vector class.
  def __init__(self, x = 0, y = 0, z = 0):
    self.x = x
    self.y = y
    self.z = z
    self.distance = sqrt(x**2 + y**2 + z**2)

def ModelToPlane(x, y, z, cz = 1, vz = 1): # Here's the magic. This basically projects the 3D model from 3D onto a 2D plane (the camera). This is how I dewarp.
  # (x, y, z) is the coordinate of the 3D point. (cz, vz) are the heights of the camera and viewer. I played with them, and they don't do much when they're fairly big.
  # I actually just translated this into Python from this Wikipedia article: http://en.wikipedia.org/wiki/3D_projection
  # But the math involved isn't that hard. It's just finding the intersection of a line and a plane, where the line is pointing at a focus point above the plane. This approach is just more general.
  a = Point(x, y, z)
  b = Point()
  c = Point(0, 0, cz)
  theta = Point(0, 0, 0)
  d = Point()
  e = Point(0, 0, vz)
  
  costx = cos(theta.x)
  costy = cos(theta.y)
  costz = cos(theta.z)
  
  sintx = sin(theta.x)
  sinty = sin(theta.y)
  sintz = sin(theta.z)
  
  d.x = costy * (sintz * (a.y - c.y) + costz * (a.x - c.x)) - sinty * (a.z - c.z)
  d.y = sintx * (costy * (a.z - c.z) + sinty * (sintz * (a.y - c.y) + costz * (a.x - c.x))) + costx * (costz * (a.y - c.y) - sintz * (a.x - c.x))
  d.z = costx * (costy * (a.z - c.z) + sinty * (sintz * (a.y - c.y) + costz * (a.x - c.x))) - sintx * (costz * (a.y - c.y) - sintz * (a.x - c.x))
  
  b.x = (d.x - e.x) * (e.z / d.z)
  b.y = (d.y - e.y) * (e.z / d.z)
  
  return (b.x, b.y) # And the coordinate of the 3D point mapped to 2D!


def RadialDewarp(image, k1, k2 = 0, k3 = 0): # This works (kind of), but it's slow and messy. OpenCV takes care of distortion, but I have yet to calibrate my camera...
  canvas = Image.new('RGB', image.size)
  draw = ImageDraw.Draw(canvas)
  image_load = image.load()
  
  center_x = image.size[0] / 2.0
  center_y = image.size[1] / 2.0
  
  for x in xrange(image.size[0]):
    for y in xrange(image.size[1]):
      r = sqrt((center_x - x)**2 + (center_y - y)**2)
      c = (1 + k1 * r**2  + k2 * r**4 + k3 * r**6)
      
      draw.point((x * c, y * c), fill = image_load[x, y])
  
  return canvas

image = Image.fromstring('RGB', cv.GetSize(imageThreshold), cv.LoadImage(imagePath).tostring()) # A copy of the image, again. I have no idea why it's here...
pixels = image.load() # More iterable pixels.
canvas = Image.new('RGB', image.size) # The output canvas.

edge_top = [] # I split the two book curves up.
edge_bottom = []

camera_offset = 20 # This is the height of the camera from the 3D model (I have no idea what the units are). Anything > 15 looks identical, so 20 works well.

for point in verticies: # Iterates through the points
  point2 = ModelToPlane(point[0], point[1], point[2], camera_offset, camera_offset + 1) # 3D -> 2D
  
  if point2[1] > 0.0: # If point falls above y = 0, it's in the top curve. If not, it's in the bottom.
    edge_top.append(point2[1])
  else:
    edge_bottom.append(point2[1])

# Normalize vertical offset of the top/bottom edge (minimum -> 0)
edge_top = [(point - min(edge_top)) for point in edge_top]
edge_bottom = [(point - min(edge_bottom)) for point in edge_bottom]

# Normalize height of the offsets (maximum -> 1)
edge_top = [point / max(edge_top) for point in edge_top]
edge_bottom = [point / max(edge_bottom) for point in edge_bottom]


# These are the ugly values. They are the y-coordinates of the book's seam, from the original picture. I have to get around this somehow...
offset_top = 340
offset_top2 = 240
offset_bottom = 240

line_top = [] # Initializes the top line of the book. This is the page's top curve.
line_bottom = []

for x in xrange(0, image.size[0] - 1):
  line_bottom.append(image.size[1] - offset_bottom + (edge_top[int(x / 2)] * offset_bottom) / sqrt(2)) # edge_top is normalized, so it is just a function going from 0 -> 1. I multiply it around, and it gets me the shape of the top of the book.
  line_top.append(offset_top - offset_top2 + (edge_bottom[int(x / 2)] * offset_top) / sqrt(2))

# Debug lines
for x in xrange(0, image.size[0] - 2):
  cv.Line(cv_image, (x, top[x]), (x + 1, top[x + 1]), cv.RGB(255, 0, 255), thickness = 2)
  cv.Line(cv_image, (x, bottom[x]), (x + 1, bottom[x + 1]), cv.RGB(255, 0, 0), thickness = 2)
  
  cv.Line(cv_image, (x, line_top[x]), (x + 1, line_top[x + 1]), cv.RGB(0, 255, 0), thickness = 2)
  cv.Line(cv_image, (x, line_bottom[x]), (x + 1, line_bottom[x + 1]), cv.RGB(0, 0, 255), thickness = 2)

# The actual dewarping
for x in xrange(0, image.size[0] - 1): # Loops through columns of pixels.
  column = image.crop((x, line_top[x], x + 1, line_bottom[x])) # Crops out a pixel column which falls between the top and bottom book shape lines.
  column = column.resize((1, image.size[1])) # Normalizes the height.
  
  canvas.paste(column, (x, 0, x + column.size[0], column.size[1])) # Pastes it onto the canvas, column by column.

canvas = canvas.resize((canvas.size[0], int(canvas.size[1] / sqrt(2)))) # This fixes the 45 degree platen, for now. When I have a minute, I will calculate the actual width of the book and use that, as this makes things look odd for books without a platen.

canvas.save('canvas.png', 'png')

cv.ShowImage('Test', cv_image)
cv.WaitKey(0)

cv.ShowImage('Test', cv.LoadImage('canvas.png'))
cv.WaitKey(0)
