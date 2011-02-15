import os, sys
from math import *

import cv
import Image, ImageOps, ImageDraw
import Gnuplot
import numpy

def numpy_smooth(x, window_len = 11, window = 'hanning'):
  if x.ndim != 1:
    raise ValueError, "smooth only accepts 1 dimensional arrays."

  if x.size < window_len:
    raise ValueError, "Input vector needs to be bigger than window size."
  
  if window_len < 3:
    return x

  if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    raise ValueError, "Window has to be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


  s = numpy.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
  
  if window == 'flat':
    w = numpy.ones(window_len, 'd')
  else:
    w = eval('numpy.' + window + '(window_len)')

  y = numpy.convolve(w / w.sum(), s, mode = 'same')
  
  return y[window_len: -window_len + 1]



try:
  import Blender
  import Blender.NMesh
  
  blender = True
  imagePath = '/home/mom/Desktop/CIMG0918.jpg'
except ImportError:
  blender = False
  imagePath = sys.argv[1]

image = cv.LoadImage(imagePath)
imageThreshold = cv.CreateImage(cv.GetSize(image), 8, 1)

cv.NamedWindow('Test', 0)
cv.ResizeWindow('Test', 1000, 600)

channelG = cv.CreateImage(cv.GetSize(image), 8, 1)
channelB = cv.CreateImage(cv.GetSize(image), 8, 1)

cv.Split(image, cv.CreateImage(cv.GetSize(image), 8, 1), channelG, channelB, None)
cv.Sub(channelG, channelB, imageThreshold)

cv.InRangeS(imageThreshold, cv.Scalar(24, 24, 24), cv.Scalar(255, 255, 255), imageThreshold)

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



verticies = []
faces = []

for i in xrange(0, len(top), 2):
  verticies.append([(len(top) / 2 - float(i)) / 1000.0, 0.5, float(top[i] - min(top)) / 1000.0])
  verticies.append([(len(top) / 2 - float(i)) / 1000.0, -0.5, float(max(bottom) - bottom[i]) / 1000.0])

for i in xrange(4, len(verticies), 4):
  faces.append([verticies[i - 4], verticies[i - 3], verticies[i - 2], verticies[i - 1]])

if blender:
  mesh = Blender.NMesh.GetRaw()
  mesh.verts = [Blender.NMesh.Vert(vert[0], vert[1], vert[2]) for vert in verticies]
  
  for i in xrange(3, len(mesh.verts), 1):
    bFace = Blender.NMesh.Face()
    
    for j in xrange(3):
      bFace.v.append(mesh.verts[i - j])
    
    mesh.faces.append(bFace)
  
  meshObject = Blender.NMesh.PutRaw(mesh)


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
  
  d.x = cos(theta.y) * (sin(theta.z) * (a.y - c.y) + cos(theta.z) * (a.x - c.x)) - sin(theta.y) * (a.z - c.z)
  d.y = sin(theta.x) * (cos(theta.y) * (a.z - c.z) + sin(theta.y) * (sin(theta.z) * (a.y - c.y) + cos(theta.z) * (a.x - c.x))) + cos(theta.x) * (cos(theta.z) * (a.y - c.y) - sin(theta.z) * (a.x - c.x))
  d.z = cos(theta.x) * (cos(theta.y) * (a.z - c.z) + sin(theta.y) * (sin(theta.z) * (a.y - c.y) + cos(theta.z) * (a.x - c.x))) - sin(theta.x) * (cos(theta.z) * (a.y - c.y) - sin(theta.z) * (a.x - c.x))
  
  b.x = (d.x - e.x) * (e.z / d.z)
  b.y = (d.y - e.y) * (e.z / d.z)
  
  return (b.x, b.y)



if not blender:
  image = Image.fromstring('RGB', cv.GetSize(imageThreshold), cv.LoadImage(imagePath).tostring())
  pixels = image.load()
  canvas = Image.new('RGB', image.size)
  draw = ImageDraw.Draw(canvas)
  
  points = []
  
  for point in verticies:
    point2 = ModelToPlane(point[0], point[1], point[2], 1, 2)
    
    points.append(point2)
    draw.point((points[-1][0], points[-1][1]), fill = 256)
  
  canvas.save('canvas.png', 'png')
  
  plot3 = Gnuplot.Gnuplot()
  plot3.title('3D Model')
  plot3.plot(points)
  
  plot1 = Gnuplot.Gnuplot()
  plot1.title('Top curve')
  plot1.plot(top)

  plot2 = Gnuplot.Gnuplot()
  plot2.title('Bottom curve')
  plot2.plot(bottom)

  cv.ShowImage('Test', imageThreshold)
  
  cv.WaitKey(0)
  
