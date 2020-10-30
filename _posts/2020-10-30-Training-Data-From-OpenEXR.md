---
toc: true
layout: post
description: Convert .exr outputs from Blender into COCO format training data.
categories: [Computer Vision,Blender]
image: images/2020-10-30-Training-Data-From-OpenEXR/header.png
---

Introduction
-------------

At the end of the [previous post](https://cgcooke.github.io/Blog/computer%20vision/blender/2020/10/23/Synthetic-Training-Data-With-Blender.html), I had shown how to use *Blender* to generate depth maps, and semantic segmentation maps.

However, this infomration is in [OpenEXR](https://en.wikipedia.org/wiki/OpenEXR) format, and we need to transform it to a form more suitible for training a computer vision model.

While writing this post, I found [this](http://www.tobias-weis.de/groundtruth-data-for-computer-vision-with-blender/) post by *Tobias Weis* to be very helpful.


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

### Extracting 

Next, we can use some "boilerplate" code to convert the exr file into a Numpy array.

In this case, we illo load in 

```python
def exr2numpy(exr_path):
    '''
    See:
    https://excamera.com/articles/26/doc/intro.html
    http://www.tobias-weis.de/groundtruth-data-for-computer-vision-with-blender/
    '''
    file = OpenEXR.InputFile(exr_path)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    Float_Type = Imath.PixelType(Imath.PixelType.FLOAT)
    
    datastr = file.channel('R', Float_Type)
    data = np.fromstring(datastr, dtype = np.float32).reshape(size[1],-1)
    
    return(data)

depth = exr2numpy("Metadata/Depth/Image0001.exr")
```

```python
fig = plt.figure()
plt.imshow(depth)
plt.colorbar()
plt.show()
```

### Creating bounding boxes

![_config.yml]({{ site.baseurl }}/images/2020-10-30-Training-Data-From-EXR/Figure_1.png)


```python
# Create figure and axes
fig,ax = plt.subplots(1)
# Display the image
ax.imshow(data)

for i in np.unique(data):
    #index 0 is the background
    if i!=0:
    	#Find the location of the object mask
        yi,xi = np.where(depth_data == i)

        print(i,np.min(xi),np.max(xi),np.min(yi),np.max(yi))

        # Create a Rectangle patch

        #box = 
        rect = Rectangle(np.min(xi), np.min(yi), np.max(xi)-np.min(xi), np.max(yi)-np.min(yi), linewidth=2, edgecolor='r', facecolor='none', alpha=0.8)

        # Add the patch to the Axes
        ax.add_patch(rect)

plt.show()
```

![_config.yml]({{ site.baseurl }}/images/2020-10-30-Training-Data-From-EXR/Figure_2.png)





