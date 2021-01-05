---
toc: true
layout: post
description: We use Tesseract to capture numbers from an image.
categories: [Computer Vision, Tesseract, Nuketown84]
image: images/2021-01-03-Number-Extraction-With-Tesseract/header.png
---

Introduction
-------------

While comparing two different mini-maps can tell us the change in angle/heading ($\omega$) between them, we can determine the heading of the player ($\theta$), via the compass. This can be seen at the top center of the screen (107 degrees).

![_config.yml]({{ site.baseurl }}/images/2020-12-18-A-Playground-In-Nuketown/Nuketown-84-1.jpg)


By using Optical Character Recognition (OCR), we can read this heading, allowing us to digitise the players current heading on a frame by frame basis. I've choses to use [Tesseract](https://github.com/tesseract-ocr/tesseract), a powerful open source library for OCR. In Python, we can use [Pytesseract](https://pypi.org/project/pytesseract/), which is a wrapper around *Tesseract*. 

You can find the video I'm digitising [here](https://www.youtube.com/watch?v=dozMeWeraFk).


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
Let's load the first frame:

```python
def load_frame(frame_number):
    cap.set(1, frame_number-1)
    res, frame = cap.read()
    return(frame)
```

Now let's preprocess the frame, by: 

1. Cropping the image down so that it only contains the number.
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

Once Tesseract has processed the image, we want to extract both the angle, and how confident *Tesseract* was in it's detection from the metadata. Often when Tesseract digitises a number, it will include other characters, which we need to strip out using "re.sub("[^0-9^]", "",text)".


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

Let's now extract a single frame:

```python
cap = cv2.VideoCapture('../HighRes.mp4')
img = load_frame(0,cap)
plt.imshow(img[:,:,::-1])
plt.show()
```
![_config.yml]({{ site.baseurl }}/images/2021-01-03-Number-Extraction-With-Tesseract/Frame.png)


And after preprocessing:
```python
img = preprocess_frame(img)
plt.imshow(img,cmap='gray')
plt.show()
```

![_config.yml]({{ site.baseurl }}/images/2021-01-03-Number-Extraction-With-Tesseract/Number.png)


Processing
-------------


Now that we have pre-processed our images, it's time to use Tesseract to digitise the text.


We have the opportunity to configure Tesseract, you can read more about the options available [here](https://ai-facets.org/tesseract-ocr-best-practices/).

```python
tesseract_config = r'--oem 3 --psm 13'
result_dict = pytesseract.image_to_data(img, config = tesseract_config, output_type = pytesseract.Output.DICT)
``` 

Let's now process the first 5,000 frames from the video: 


```python
angles = []
confidences = []
for i in range(0,5_000):
    img = load_frame(i,cap)
    img = preprocess_frame(img)

    tesseract_config = r'--oem 3 --psm 13'
    result_dict = pytesseract.image_to_data(img, config = tesseract_config, output_type = pytesseract.Output.DICT)
    
    angle, confidence = extract_angle_confidence(result_dict)
    
    angles.append(angle)
    confidences.append(confidence)
```

We can save both the angles, and Tesseract's level of confidence.

```python
angles = np.array(angles)
confidences = np.array(confidences)

np.save('angles.npy',angles)
np.save('confidences.npy',confidences)
```

Results Analysis

-------------

Finally, let's do some quick analysis of the data extracted.


By examining the histogram, we find that Tesseract was often uncertain about the results it was generating.

```python
ax = sns.histplot(confidences,bins = np.arange(0,100,10))
ax.set_xlabel('Confidence (%)')
```

![_config.yml]({{ site.baseurl }}/images/2021-01-03-Number-Extraction-With-Tesseract/Histogram.png)



We can also chart the angles on a frame by frame basis, note that the limits of the chart are set to the range of 0-360 (degrees).

```python
plt.plot(angles,alpha=0.5)
plt.ylim(0,360)
plt.xlabel('Sample Number')
plt.ylabel('Angle (Degrees)')
plt.show()
```

![_config.yml]({{ site.baseurl }}/images/2021-01-03-Number-Extraction-With-Tesseract/LinePlot.png)



Using:
```python
np.count_nonzero(~np.isnan(angles))
```
We find that Tesseract managed to extract numbers in 4,049 out of 5,000 frames.

Conclusion
-------------
Our next step is to take this heading data, and integrate it with other sources of data, in order to form a more coherent view of the players position and heading. We can already see that our method will need to be robust to both missing and erroneous data. 





