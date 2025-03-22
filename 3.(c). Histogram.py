# -*- coding: utf-8 -*-
"""
Created on Fri May 22 23:09:53 2020

@author: sivas
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
k="D:/LIDC-IDRI/multin/"
area_plaque=[]
hist_1=[]
x=0
for i in os.listdir(k):
    i=os.path.join(k,i)
    im=Image.open(i)
    k1=im.convert('L')
    pixels=list(k1.getdata())
    h=len(pixels)
    #print(x)
    while h>0:
        for i in pixels:
            if i!=255:
                x=x+1
                hist_1.append(i)
            
            h-=1
    #print(x)
    area_plaque.append(x)
    x=0
print(area_plaque)
print(sum(area_plaque))
#print(hist_1)
plt.hist(hist_1)
import numpy
import matplotlib.pyplot as plt

x = hist_1
min_r=min(hist_1)
max_r=max(hist_1)
range_1= max_r - min_r
#print(range_1)

values, bins, _ = plt.hist(x,color='lightgrey', edgecolor='black',
              linewidth=1)
#plt.ylim(0,20000000)
plt.xlabel("Grayscale value")
plt.ylabel("Frequency (x1e7)")
#print(bins,values)
a_h=[]
y=bins.tolist()
z=values.tolist()
#print(y)

diff_interval = [y[i + 1] - y[i] for i in range(len(y)-1)] 
#print(diff_interval)
# frequency densit
fd=[]
for i in range(len(z)):
    fd.append(z[i]/diff_interval[i])
#print(fd)
u=0
for i in range(len(fd)):
    u+=fd[i]*diff_interval[i]
print(u)
#print(fd,diff_interval)
# calculation of A1/B1
A1=0
for i in range(8):
    A1+=fd[i]*diff_interval[i]
print(A1)
# calculation of A2
A2=0
for i in range(1,3):
    A2+=fd[-i]*diff_interval[-i]
print(A2)    
    
    