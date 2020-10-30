---
toc: true
layout: post
description: Let's learn how we can create depth and semantic maps, for training machine learning models.
categories: [Computer Vision,Blender]
image: images/2020-10-23-Synthetic-Training-Data-With-Blender/header.png
---

Introduction
-------------

While writingthis post, I found [this](http://www.tobias-weis.de/groundtruth-data-for-computer-vision-with-blender/) post by *Tobias Weis* to be really helpful for understanding rendering nodes. 


The Code
-------------


```python
import OpenEXR
import Imath
import array
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
```

```python
def exr2numpy(exr):
    #http://www.tobias-weis.de/groundtruth-data-for-computer-vision-with-blender/
    """ converts 1-channel exr-data to 2D numpy arrays """                                                                    
    file = OpenEXR.InputFile(exr)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    
    (R) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R") ]

    data = np.array(R).reshape(sz[1],-1)
    return data

depth_data = exr2numpy("Metadata/Index/Image0001.exr")

fig = plt.figure()
plt.imshow(depth_data)
plt.colorbar()
plt.show()
```


```python

# Create figure and axes
fig,ax = plt.subplots(1)
# Display the image
ax.imshow(depth_data)

for i in np.unique(depth_data):
    #index 0 is the background
    if i!=0:

    	#Find the location of the object mask
        yi,xi = np.where(depth_data == i)

        print(i,np.min(xi),np.max(xi),np.min(yi),np.max(yi))

        # Create a Rectangle patch
        rect = Rectangle((np.min(xi), np.min(yi)), np.max(xi)-np.min(xi), np.max(yi)-np.min(yi), linewidth=2, edgecolor='r', facecolor='none', alpha=0.8)

        # Add the patch to the Axes
        ax.add_patch(rect)


plt.show()
```

![_config.yml]({{ site.baseurl }}/images/2020-10-30-Training-Data-From-EXR/Figure_1.png)

![_config.yml]({{ site.baseurl }}/images/2020-10-30-Training-Data-From-EXR/Figure_2.png)



