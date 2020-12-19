import math
import numpy as np
import platform
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
# necessite scikit-image 
from skimage import io as skio
#########################################################


#%%
plt.set_cmap('gray')
 
sh=[128,128]

im_bin=2-np.ceil(2*np.random.rand(*sh))
plt.imshow(im_bin);

import echan
#%%
#definir la valeur de beta pour le champ que vous voulez simuler
beta_reg=-40

mafigure=plt.figure()
plt.imshow(im_bin);        
mafigure.canvas.draw()
plt.show()
#test=plt.waitforbuttonpress()

for n in range(30):  
    print(n)
    im_bin = echan.echan(im_bin,beta_reg) 
    plt.imshow(im_bin);        
    mafigure.canvas.draw()
    plt.show(block=False)
    #test=plt.waitforbuttonpress()

    
plt.figure()
plt.imshow(im_bin);
plt.show()
#%%
