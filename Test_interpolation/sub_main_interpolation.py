### I try to use scipy.interpolate.interp2d to replace BBS. But I find its performance is really bad.

import scipy.io
import scipy.interpolate
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm


filename='D:/MATLAB/Deep_learnning_based_conformal_NRSfM/Data_generation/text_intorpolation/matlab.mat'
mat = scipy.io.loadmat(filename)
Point_3d = mat['Point_3d']
Point_3d_sorted= Point_3d#[:,Point_3d[0,:].argsort()]
x=Point_3d_sorted[0,:]
y=Point_3d_sorted[1,:]
z=Point_3d_sorted[2,:]
umax = max(x)
umin = min(x)
vmax = max(y)
vmin = min(y)
n=99
dx=(umax-umin)/(n+1)
dy=(vmax-vmin)/(n+1)
xnew = np.arange(umin, umax, dx)
ynew = np.arange(vmin, vmax, dy)
xx, yy = np.meshgrid(xnew, ynew)
xx=xx.flatten()
yy=yy.flatten()
f = scipy.interpolate.interp2d(x, y, z, kind='cubic')
for i in range(xx.shape[0]):
    zz_v=f(xx[i],yy[i])
#mydata=np.concatenate((xnew, ynew,zz_v))
#scipy.io.savemat('test.mat', {'mydata': mydata})
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xnew,ynew, zz_v, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
