import os, sys
from math import *

import cv
import Image, ImageOps, ImageDraw
import Gnuplot
import numpy

def numpy_smooth(x, window_len = 11, window = 'hanning'):
  if window_len < 3:  return x
  
  if window == 'flat':
    w = numpy.ones(window_len, 'd')
  else:
    w = eval('numpy.' + window + '(window_len)')
  
  return numpy.convolve(w / w.sum(), numpy.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]], mode = 'same')[window_len: -window_len + 1]



imagePath = sys.argv[1]

image = cv.LoadImage(imagePath)
cv_image = image
cv_blank = cv.CreateImage(cv.GetSize(image), 8, 1)

imageThreshold = cv.CreateImage(cv.GetSize(image), 8, 1)

cv.NamedWindow('Test', 0)
cv.ResizeWindow('Test', 1000, 600)

channelG = cv.CreateImage(cv.GetSize(image), 8, 1)
channelB = cv.CreateImage(cv.GetSize(image), 8, 1)

cv.Split(image, cv.CreateImage(cv.GetSize(image), 8, 1), channelG, channelB, None)

cv.Sub(channelG, channelB, imageThreshold)

cv.InRangeS(imageThreshold, cv.Scalar(24, 24, 24), cv.Scalar(255, 255, 255), imageThreshold)

cv.ShowImage('Test', imageThreshold)
cv.WaitKey(0)

cv.Smooth(imageThreshold, imageThreshold, smoothtype = cv.CV_MEDIAN, param1 = 9)

cv.ShowImage('Test', imageThreshold)
cv.WaitKey(0)

image = Image.fromstring('L', cv.GetSize(imageThreshold), imageThreshold.tostring())
pixels = image.load()



top = []
bottom = []

for x in xrange(image.size[0]):
  intersects = []
  
  for y in reversed(xrange(image.size[1] / 2, image.size[1])):
    if pixels[x, y] == 255:
      intersects.append(y)
    
  if len(intersects) > 1:
    top.append((intersects[-1] + intersects[0]) / 2)
  elif len(top) > 0:
    top.append(top[-1])
  else:
    top.append(3 * image.size[1] / 4)

for x in xrange(image.size[0]):
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


#top = numpy_smooth(numpy.array(top), window_len = 51)
#bottom = numpy_smooth(numpy.array(top), window_len = 51)


verticies = []

for i in xrange(0, len(top), 2):
  verticies.append([(len(top) / 2 - float(i)) / 1000.0, 0.5, float(top[i] - min(top)) / 1000.0])
  verticies.append([(len(top) / 2 - float(i)) / 1000.0, -0.5, float(max(bottom) - bottom[i]) / 1000.0])


class Point:
  def __init__(self, x = 0, y = 0, z = 0):
    self.x = x
    self.y = y
    self.z = z
    self.distance = sqrt(x**2 + y**2 + z**2)

def ModelToPlane(x, y, z, cz = 1, vz = 1):
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
  
  return (b.x, b.y)

def RadialDewarp(image, k1, k2 = 0, k3 = 0):
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

image = Image.fromstring('RGB', cv.GetSize(imageThreshold), cv.LoadImage(imagePath).tostring())
pixels = image.load()
canvas = Image.new('RGB', image.size)
draw = ImageDraw.Draw(canvas)

edge_top = []
edge_bottom = []

if len(sys.argv) > 2:
  camera_offset = int(sys.argv[2])
else:
  camera_offset = 20

for point in verticies:
  point2 = ModelToPlane(point[0], point[1], point[2], camera_offset, camera_offset + 1)
  
  if point2[1] > 0.0:
    edge_top.append(point2[1])
  else:
    edge_bottom.append(point2[1])

# Normalize offset (min = 0)
edge_top = [(point - min(edge_top)) for point in edge_top]
edge_bottom = [(point - min(edge_bottom)) for point in edge_bottom]

# Normalize height (max = 1)
edge_top = [point / max(edge_top) for point in edge_top]
edge_bottom = [point / max(edge_bottom) for point in edge_bottom]

offset_top = 109
offset_top2 = 75
offset_bottom = 75

line_top = []
line_bottom = []

for x in xrange(0, image.size[0] - 1):
  line_bottom.append(image.size[1] - offset_bottom + (edge_top[int(x / 2)] * offset_bottom) / sqrt(2))
  line_top.append(offset_top - offset_top2 + (edge_bottom[int(x / 2)] * offset_top) / sqrt(2))

for x in xrange(0, image.size[0] - 2):
  cv.Line(cv_image, (x, top[x]), (x + 1, top[x + 1]), cv.RGB(255, 0, 255), thickness = 2)
  cv.Line(cv_image, (x, bottom[x]), (x + 1, bottom[x + 1]), cv.RGB(255, 0, 0), thickness = 2)
  
  cv.Line(cv_image, (x, line_top[x]), (x + 1, line_top[x + 1]), cv.RGB(0, 255, 0), thickness = 2)
  cv.Line(cv_image, (x, line_bottom[x]), (x + 1, line_bottom[x + 1]), cv.RGB(0, 0, 255), thickness = 2)

for x in xrange(0, image.size[0] - 1):
  column = image.crop((x, line_top[x], x + 1, line_bottom[x]))
  column = column.resize((1, image.size[1]))
  
  canvas.paste(column, (x, 0, x + column.size[0], column.size[1]))

canvas = canvas.resize((canvas.size[0], int(canvas.size[1] / sqrt(2))))

if len(sys.argv) > 3:
  canvas.save(sys.argv[3], 'png')
else:
  canvas.save('canvas.png', 'png')

cv.ShowImage('Test', cv_image)
cv.WaitKey(0)

cv.ShowImage('Test', cv.LoadImage('canvas.png'))
cv.WaitKey(0)

'''
plot1 = Gnuplot.Gnuplot()
plot1.title('Top curve')
plot1.plot(edge_top)

plot2 = Gnuplot.Gnuplot()
plot2.title('Bottom curve')
plot2.plot(edge_bottom)

cv.ShowImage('Test', imageThreshold)

cv.WaitKey(0)

cv.ShowImage('Test', cv_image)

cv.WaitKey(0)
'''
