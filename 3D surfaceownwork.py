# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:44:25 2020

@author: sivas
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.title("Accuracy Graph")


Y = [   
            
            
            [19, 19, 19, 19, 19, 19],
            [17, 17, 17, 17, 17, 17],
            [15, 15, 15, 15, 15, 15],
            [13, 13, 13, 13, 13, 13],
            [11, 11, 11, 11, 11, 11],
            [9, 9, 9, 9, 9, 9],
            [7, 7, 7, 7, 7, 7],
            [5, 5, 5, 5, 5, 5],
            
            
            
            
            
            
            
            
            
        ]
X = [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6], 
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            
        ]

Z = [
            
         
            
            [78.85,79.58,84.45,84.45,91.604,87.57],
            [70.54,79.85,87.85,85.85,91.468,88.85],
            [72.52,78.54,82.52,91.11,95.801,89.85],
            [71.15,80.54,84.45,91.21,93.843,91.02],
            [73.35,81.12,85.56,90.54,94.553,91.54],
            [72.54,81.54,82.52,91.51,93.853,90.25],
            [64.45,75.45,78.85,84.45,88.05,82.45],
            [58.85,62.52,68.85,72.52,76.204,70.54],

            
              
             
]
my_cmap = plt.get_cmap('hot') 
Y=np.array(Y)

ax.plot_surface(X, Y, Z, rstride=1, cmap=cm.coolwarm, edgecolor='none',cstride=1)
#ax.plot_wireframe(X, Y, Z,cmap=cm.jet)
#ax.bar3d(X, Y, Z,x_size, y_size, z_size,cmap=cm.jet)
#ax.contour3D(X, Y, Z,  cmap=cm.jet)
X=np.array(X)
Y=np.array(Y)
Z=np.array(Z)
print(type(X))
print(X)
ax.set_xlabel('Augmentation',fontname='Times New Roman', fontsize=11)
ax.set_ylabel('CNN Layers',fontname='Times New Roman', fontsize=11)
ax.set_zlabel('Accuracy (%)',fontname='Times New Roman', fontsize=11)
#ax.set_yticks([5,7,9,11,13,15])
ax.set_yticks([19,17,15,13,11,9,7,5])
ax.set_ylim(15,5)
ax.set_xticks([6,5,4,3,2,1])
ax.set_zlim(55,96)
ax.invert_xaxis()
ax.view_init(elev=38, azim=120)
#plt.savefig("3d_test2.png",dpi=700)
#cset = ax.contourf(X, Y, Z,                    zdir ='z',                    offset = np.min(Z),    cmap = cm.jet               ) 
#cset = ax.contourf(X, Y, Z,                    zdir ='x',                    offset =-5,                    cmap = my_cmap) 
#cset = ax.contourf(X, Y, Z,                     zdir ='y',                    offset = 5,                    cmap = my_cmap) 
#fig.colorbar(surf, ax = ax,               shrink = 0.5,              aspect = 5) 
#x = np.arange(1, 10)
#y = x.reshape(-1, 1)

#ax.scatter(np.array([6.1]),           np.array([10.5]),            np.array([97.54]),           color='blue',           s=40        )


"""
# Set rotation angle to 60 degrees
ax.view_init(azim=60)
ax.set_zlabel('Accuracy (%)')
ax.set_zlim(80, 100)

ax.plot(np.array([11]),
           np.array([3]),
           np.array([95.66]), 
           color='blue'
        )
"""
#ax.set_title('3-D Optimization',fontname='Times New Roman', fontsize=11);
plt.savefig('Accuracy_graph.jpg',dpi=500)
plt.show()

