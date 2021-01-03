---
toc: true
layout: post
description: We use Tesseract to capture numbers from an image.
categories: [Computer Vision, Tesseract]
image: images/2021-01-03-Number-Extraction-With-Tesseract/header.png
---

Introduction
-------------


[Tesseract](https://github.com/tesseract-ocr/tesseract)


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pytesseract
import re 

plt.rcParams['figure.figsize'] = [10, 10]
```


```python
def load_frame(frame_number):
    cap.set(1, frame_number-1)
    res, frame = cap.read()
    return(frame)

def preprocess_frame(img):
    img = img[10:50,610:670,::-1]
    img = cv2.resize(img,(600,400))
    img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return(img)
```


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
                
    return(angle,confidence)            
```



```python
fig = plt.figure()
plt.imshow(depth)
plt.colorbar()
plt.show()
```

![_config.yml]({{ site.baseurl }}/images/2021-01-03-Number-Extraction-With-Tesseract/Histogram.png)

![_config.yml]({{ site.baseurl }}/images/2021-01-03-Number-Extraction-With-Tesseract/LinePlot.png)



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

Data Aanalysis
-------------

```python
ax = sns.histplot(confidences,bins = np.arange(0,100,10))
ax.set_xlabel('Confidence (%)')
```

```python
plt.plot(angles,alpha=0.5)
plt.ylim(0,360)
plt.xlabel('Sample Number')
plt.ylabel('Angle (Degrees)')
plt.show()
```

```python
np.count_nonzero(~np.isnan(angles))
```



Conclusion
-------------






