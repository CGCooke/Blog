---
toc: true
layout: post
description: We use Tesseract to capture numbers from an image.
categories: [Computer Vision, Tesseract]
image: images/2021-01-03-Number-Extraction-With-Tesseract/header.png
---

Introduction
-------------

While comparing two different mini-maps can tell us the change in angle/heading ($\omega) between them, we can determine the heading of the player ($\omega), via the compass. This can be seen at the top center of the screen (107 degrees).

![_config.yml]({{ site.baseurl }}/images/2020-12-18-A-Playground-In-Nuketown/Nuketown-84-1.jpg)



By using Optical Character Recognition (OCR), we can read this heading, allowing us to digitise the players current heading on a frame by frame basis. I've choses to use[Tesseract](https://github.com/tesseract-ocr/tesseract), a powerful opensource library of OCR. 


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pytesseract
import re 

plt.rcParams['figure.figsize'] = [10, 10]
```

Preprocessing
-------------
Let's load the image the frame,

```python
def load_frame(frame_number):
    cap.set(1, frame_number-1)
    res, frame = cap.read()
    return(frame)
```

Now let's preprocess the frame, by 

1. Cropping the image down so that it only contain the number.
2. Resizing the image so that it's larger.
3. Converting the image to grayscale, and then inverting.

Each of these steps helps improve the accuracy of *Tesseract*.

```python

def preprocess_frame(img):
    img = img[10:50,610:670,::-1]
    img = cv2.resize(img,(600,400))
    img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return(img)
```


Once Tesseract has processed the iamge, we want to extract both the angle, and how confident *Tesseract* was from the nmetadata.
```python
def extract_angle_confidence(result_dict):
    angle = np.NaN
    confidence = np.NaN
    for i in range(0,len(result_dict['text'])):
        confidence = int(result_dict['conf'][i])
        if confidence > 0:
            text = result_dict['text'][i]
            text = re.sub("[^0-9^]", "",text)
            if len(text)>0:
                angle = int(text)
                
    return(angle, confidence)            
```



```python
fig = plt.figure()
plt.imshow(depth)
plt.colorbar()
plt.show()
```



```python
cap = cv2.VideoCapture('../HighRes.mp4')

img = load_frame(0,cap)
plt.imshow(img[:,:,::-1])
plt.show()
```
![_config.yml]({{ site.baseurl }}/images/2021-01-03-Number-Extraction-With-Tesseract/Frame.png)

```python
img = preprocess_frame(img)
plt.imshow(img,cmap='gray')
plt.show()
```

![_config.yml]({{ site.baseurl }}/images/2021-01-03-Number-Extraction-With-Tesseract/Number.png)



Processing
-------------

```python
angles = []
confidences = []
for i in range(0,5_000):
    img = load_frame(i,cap)
    img = preprocess_frame(img)

    custom_config = r'--oem 3 --psm 13'
    result_dict = pytesseract.image_to_data(img, config = custom_config, output_type = pytesseract.Output.DICT)
    
    angle, confidence = extract_angle_confidence(result_dict)
    
    angles.append(angle)
    confidences.append(confidence)


angles = np.array(angles)
confidences = np.array(confidences)

np.save('angles.npy',angles)
np.save('confidences.npy',confidences)
```

Results Analysis
-------------

```python
ax = sns.histplot(confidences,bins = np.arange(0,100,10))
ax.set_xlabel('Confidence (%)')
```


![_config.yml]({{ site.baseurl }}/images/2021-01-03-Number-Extraction-With-Tesseract/Histogram.png)



```python
plt.plot(angles,alpha=0.5)
plt.ylim(0,360)
plt.xlabel('Sample Number')
plt.ylabel('Angle (Degrees)')
plt.show()
```

![_config.yml]({{ site.baseurl }}/images/2021-01-03-Number-Extraction-With-Tesseract/LinePlot.png)



```python
np.count_nonzero(~np.isnan(angles))
```



Conclusion
-------------






