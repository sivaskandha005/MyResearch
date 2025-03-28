#!/usr/bin/env python

from __future__ import division
import numpy as np
from scipy.linalg import hankel
from scipy.signal import convolve2d
import scipy.io as sio
import matplotlib as m
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from tools import *

B = []


def bispectrumd(y, nfft=None, wind=None, nsamp=100, overlap=None):
  """
  Parameters:
    y    - data vector or time-series
    nfft - fft length [default = power of two > segsamp]
    wind - window specification for frequency-domain smoothing
           if 'wind' is a scalar, it specifies the length of the side
              of the square for the Rao-Gabr optimal window  [default=5]
           if 'wind' is a vector, a 2D window will be calculated via
              w2(i,j) = wind(i) * wind(j) * wind(i+j)
           if 'wind' is a matrix, it specifies the 2-D filter directly
    segsamp - samples per segment [default: such that we have 8 segments]
            - if y is a matrix, segsamp is set to the number of rows
    overlap - percentage overlap [default = 50]
            - if y is a matrix, overlap is set to 0.

  Output:
    Bspec   - estimated bispectrum: an nfft x nfft array, with origin
              at the center, and axes pointing down and to the right.
    waxis   - vector of frequencies associated with the rows and columns
              of Bspec;  sampling frequency is assumed to be 1.
  """

  #print(y.shape)
  y = y.flatten()
  y = y.reshape(len(y), 1)
  (ly, nrecs) = y.shape
  if ly == 1:
    y = y.shape(1,-1)
    ly = nrecs
    nrecs = 1

  
  if not nfft: nfft = 128
  if not overlap: overlap = 50
  overlap = min(99, max(overlap, 0))
  if nrecs > 1: overlap = 0
  if not nsamp: nsamp = 0
  if nrecs > 1: nsamp = ly
  if nrecs == 1 and nsamp <= 0:
    nsamp = np.fix(ly/ (8 - 7 * overlap/100))
  if nfft < nsamp:
    nfft = 2**nextpow2(nsamp)
  overlap = np.fix(nsamp*overlap / 100)
  nadvance = nsamp - overlap
  nrecs = np.fix((ly*nrecs - overlap) / nadvance)

  if not wind: wind = 5

  m = n = 0
  try:
    (m, n) = wind.shape
  except ValueError:
    (m,) = wind.shape
    n = 1
  except AttributeError:
    m = n = 1

  window = wind
  # scalar: wind is size of Rao-Gabr window
  if max(m, n) == 1:
    winsize = wind
    if winsize < 0: winsize = 5 # the window size L
    winsize = winsize - (winsize%2) + 1 # make it odd
    if winsize > 1:
      mwind = np.fix(nfft/winsize) # the scale parameter M
      lby2 = (winsize - 1)/2

      theta = np.array([np.arange(-1*lby2, lby2+1)]) # force a 2D array
      #print(theta)
      opwind = np.ones([winsize, 1]) * (theta**2) # w(m,n) = m**2
      opwind = opwind + opwind.transpose() + (np.transpose(theta) * theta) # m**2 + n**2 + mn
      opwind = 1 - ((2*mwind/nfft)**2) * opwind
      Hex = np.ones([winsize,1]) * theta
      Hex = abs(Hex) + abs(np.transpose(Hex)) + abs(Hex + np.transpose(Hex))
      Hex = (Hex < winsize)
      opwind = opwind * Hex
      opwind = opwind * (4 * mwind**2) / (7 * np.pi**2)
    else:
      opwind = 1

  # 1-D window passed: convert to 2-D
  elif min(m, n) == 1:
    window = window.reshape(1,-1)

    if np.any(np.imag(window)) != 0:
      #print( "1-D window has imaginary components: window ignored")
      window = 1

    if np.any(window) < 0:
      #print ("1-D window has negative components: window ignored")
      window = 1

    lwind = np.size(window)
    w = window.ravel(order='F')
    # the full symmetric 1-D
    windf = np.array(w[range(lwind-1, 0, -1) + [window]])
    window = np.array([window], np.zeros([lwind-1,1]))
    # w(m)w(n)w(m+n)
    opwind = (windf * np.transpose(windf)) * hankel(np.flipud(window), window)
    winsize = np.size(window)

  # 2-D window passed: use directly
  else:
    winsize = m

    if m != n:
      #print ("2-D window is not square: window ignored")
      window = 1
      winsize = m

    if m%2 == 0:
      #print ("2-D window does not have odd length: window ignored")
      window = 1
      winsize = m

    opwind = window


  # accumulate triple products
  Bspec = np.zeros([nfft, nfft]) # the hankel mask (faster)
  A = []
  for index in range(nfft-1):
      A.append(index)

  mask = hankel(np.arange(nfft),np.array([nfft-1]+A))
  locseg = np.arange(nsamp).transpose()
  y = y.ravel(order='F')

  for krec in range(int(nrecs)):
    xseg = y[locseg].reshape(1,-1)
    Xf = np.fft.fft(xseg - np.mean(xseg), nfft) / nsamp
    CXf = np.conjugate(Xf).ravel(order='F')
    Bspec = Bspec + \
      flat_eq(Bspec, (Xf * np.transpose(Xf)) * CXf[mask].reshape(nfft, nfft))
    locseg = locseg + int(nadvance)

  Bspec = np.fft.fftshift(Bspec) / nrecs


  # frequency-domain smoothing
  if winsize > 1:
    lby2 = int((winsize-1)/2)
    Bspec = convolve2d(Bspec,opwind)
    Bspec = Bspec[range(lby2+1,lby2+nfft+1), :][:, np.arange(lby2+1,lby2+nfft+1)]


  if nfft%2 == 0:
    waxis = np.transpose(np.arange(-1*nfft/2, nfft/2)) / nfft
  else:
    waxis = np.transpose(np.arange(-1*(nfft-1)/2, (nfft-1)/2+1)) / nfft


  #cont1 = plt.contour(abs(Bspec), 4, waxis, waxis)
  

  #cont = plt.contourf(waxis, waxis, abs(Bspec), 100, cmap=plt.cm.Spectral_r)
  #plt.colorbar(cont)
  #plt.title('Bispectrum estimated via the direct (FFT) method')
  #plt.xlabel('f1')
  #plt.ylabel('f2')
  #plt.show()

  #X,Y = np.meshgrid(waxis,waxis)
  #ax = plt.axes(projection='3d')
  #cont = ax.plot_surface(X, Y, abs(Bspec), rstride=10, cstride=10,cmap='viridis', edgecolor='none')
  #plt.colorbar(cont)
  #plt.xlabel('f1')
  #plt.ylabel('f2')
  #ax.set_zlabel('B-Value')
  #ax.set_zlim(0, 1000)
  #ax.set_title('Bi-Spectrum');
  #plt.show()

  
  B2 = []
  for l in range(len(Bspec)):
      B2.append(abs(Bspec[l]))

  #print(Bspec.mean())
  B.append(np.sum(B2)/len(B2))
  return (Bspec, waxis)

def test(y):
  #qpc = sio.loadmat(here(__file__) + '/Normal_0.mat')
  dbic, waxis1 = bispectrumd(y, 128,3,64,0)
  return (dbic, waxis1)
  #print(B3)
import os
import cv2
test1= "D:/LIDC-IDRI/fin/malgin/"
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, iradon
from scipy.ndimage import zoom

B3 = []
waxis1 = 0

index = 0
for i in os.listdir(test1):
    if i.endswith('jpg'):
        k=os.path.join(test1,i)
        image = imread(k, as_grey=True)
        projections = radon(image/255., theta=[0])
        dbic, waxis1 =test(projections)
        if(index ==0):
            B3 = abs(dbic)
        else:
            B3 += abs(dbic)
        index += 1

#print(B)
A1 = []
for i in range (len(B)):
    A1.append(abs(B[i]))
    
    
print(np.sum(A1)/len(A1))
print(B3)
X,Y = np.meshgrid(waxis1,waxis1)

Bspec = B3/len(A1)
#Bspec = Bspec/len(A1)
#print(Bspec)


#plt.clf()
#plt.pcolor(X, Y, v, cmap=cm)
levels = np.linspace(0,25000,25000)
plt.contourf(waxis1, waxis1, abs(Bspec), 1000, levels=levels, cmap='viridis')
#plt.clim(0,1000)
plt.colorbar()
plt.title('Bispectrum estimated via the direct (FFT) method')
plt.xlabel('f1')
plt.ylabel('f2')
plt.show()

ax = plt.axes(projection='3d')

#plt.clf()
#plt.pcolor(X, Y, v, cmap=cm)

cont = ax.plot_surface(X, Y, abs(Bspec), rstride=1, cstride=1,
              cmap='viridis', vmin=0, vmax=2500)
plt.colorbar(cont)
plt.xlabel('f1')
plt.ylabel('f2')
ax.set_zlabel('B-Value')
#ax.set_zlim(0, 2500)
ax.set_title('Bi-Spectrum');
plt.show()




