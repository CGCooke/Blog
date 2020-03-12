---
toc: true
layout: post
description: Visualizing Digital Elevation Maps
categories: [Remote Sensing]
---

## Basic setup

On February 22, 2000, after 11 days of measurements, the most comprehensive map ever created of the earth's topography was complete. The space shuttle *Endeavor* had just completed the Shuttle Radar Topography Mission, using a specialised radar to image the earths surface.

The Digital Elevation Map (DEM) produced by this mission is in the public domain and provides the measured terrain high at ~90 meter resolution. The mission mapped 99.98% of the area between 60 degrees North and 56 degrees South.  

In this post, I will examine how to process the raw DEM so it is more intuitively interpreted, through the use of *hillshading*,*slopeshading* & *hypsometric tinting*. 


The process of transforming the raw GeoTIFF into the final imagery product is simple. Much of the grunt work being carried out by GDAL, the Geospatial Data Abstraction Library. 

In order, we need to:

1. Download a DEM as a GeoTIFF
2. Extract a subsection of the GeoTIFF
3. Reproject the subsection
4. Make an image by hillshading
5. Make an image by coloring the subsection according to altitude
6. Make an image by coloring the subsection according to slope
7. Combine the 3 images into a final composite


## DEM
===============

Several different DEM's have been created from the data collected on the SRTM mission, in this post I will use the CGIAR [SRTM 90m Digital Elevation Database](http://www.cgiar-csi.org/data/srtm-90m-digital-elevation-database-v4-1). Data is provided in 5x5 degree tiles, with each degree of latitude equal to approximately 111Km. 

Our first task is to acquire a tile. Tiles can be downloaded from http://data.cgiar-csi.org/srtm/tiles/GeoTIFF/ using wget. 




```python
import os
import math
from PIL import Image, ImageChops, ImageEnhance
from matplotlib import cm
```


```python
def downloadDEMFromCGIAR(lat,lon):
    ''' Download a DEM from CGIAR FTP repository '''
    fileName = lonLatToFileName(lon,lat)+'.zip'

    ''' Check to see if we have already downloaded the file '''
    if fileName not in os.listdir('.'):
        os.system('''wget --user=data_public --password='GDdci' http://data.cgiar-csi.org/srtm/tiles/GeoTIFF/'''+fileName)
    os.system('unzip '+fileName)
```


```python
def lonLatToFileName(lon,lat):
    ''' Compute the input file name '''
    tileX = int(math.ceil((lon+180)/5.0))
    tileY = -1*int(math.ceil((lat-65)/5.0))
    inputFileName = 'srtm_'+str(tileX).zfill(2)+'_'+str(tileY).zfill(2)
    return(inputFileName)

```


```python
lon,lat = -123,49
inputFileName = lonLatToFileName(lon,lat)
downloadDEMFromCGIAR(lat,lon)
```


## Footnotes
I found the following sources to be invaluable in compiling this post:

* [Creating color relief and slope shading](http://blog.thematicmapping.org/2012/06/creating-color-relief-and-slope-shading.html)
* [A workflow for creating beautiful relief shaded DEMs using gdal](http://linfiniti.com/2010/12/a-workflow-for-creating-beautiful-relief-shaded-dems-using-gdal/)
* [Shaded relief map in python](http://www.geophysique.be/2014/02/25/shaded-relief-map-in-python/)
* [Stamen Design](http://openterrain.tumblr.com/)

