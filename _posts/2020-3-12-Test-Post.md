---
toc: true
layout: post
description: Visualizing Digital Elevation Maps
categories: [Remote Sensing]
---
# Example Markdown Post

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


## Basic formatting

You can use *italics*, **bold**, `code font text`, and create [links](https://www.markdownguide.org/cheat-sheet/). Here's a footnote [^1]. Here's a horizontal rule:

---

## Lists

Here's a list:

- item 1
- item 2

And a numbered list:

1. item 1
1. item 2

## Boxes and stuff

> This is a quotation

{% include alert.html text="You can include alert boxes" %}

...and...

{% include info.html text="You can include info boxes" %}

## Images

![]({{ site.baseurl }}/images/logo.png "fast.ai's logo")

## Code

You can format text and code per usual 

General preformatted text:

    # Do a thing
    do_thing()

Python code and output:

```python
# Prints '2'
print(1+1)
```

    2

Formatting text as shell commands:

```shell
echo "hello world"
./some_script.sh --option "value"
wget https://example.com/cat_photo1.png
```

Formatting text as YAML:

```yaml
key: value
- another_key: "another value"
```


## Tables

| Column 1 | Column 2 |
|-|-|
| A thing | Another thing |


## Tweetcards

{% twitter https://twitter.com/jakevdp/status/1204765621767901185?s=20 %}


## Footnotes



[^1]: This is the footnote.

