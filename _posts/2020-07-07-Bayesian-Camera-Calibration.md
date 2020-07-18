---
toc: true
layout: post
description: Let's apply PyMC3 to our camera calibration problem
categories: [Bayesian, PyMC3, Computer Vision]
image: images/2020-07-07-Bayesian-Camera-Calibration/header.jpg
---

## The Context
In a previous [post](https://cgcooke.github.io/Blog/computer%20vision/optimisation/linear%20algebra/2020/02/23/An-Adventure-In-Camera-Calibration.html), I attempted to reverse-engineer information about a camera, from a photo it had taken of a scene. While I found a solution, I wasn't sure how confident I could be in the answer. I was also curious if I could improve the solution by injecting prior knowlege, from other sources.


I the idea to apply Bayesian analysis, and try and find a solution using *Makov Chain Monte Carlo* and [PyMC3](https://docs.pymc.io/). After a bit of searching, I also found this [paper](https://www.sciencedirect.com/science/article/pii/S0924271619302734), which told me that the idea wasn't completely outlandish.


In this post, we will combine a *prior* belief (probability distributions) about some of the camera's parameters, with measured 2D-3D scene correspondances. By combining these two sources of information, we can compute *posterior* distributions for each camera parameter. 

Because we have a probability distribution, we can understand how certain we are about each paramter. 

Let's start by building a model.


## Modelling
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore')

plt.rcParams['figure.figsize'] = [10,10]
```

```python
df = pd.read_csv('data/2020-07-05-Bayesian-Camera-Calibration/points.csv',sep =' ')
px = df.i.values
py = df.j.values

X_input = df.X.values
Y_input = df.Y.values
Z_input = df.Z.values

number_points = px.shape[0]

points3d = np.vstack([X_input,Y_input,Z_input]).T
```


Ok, so now that we have loaded in our 2D and 3D point correspondances, we can now turn to representing the camera itself.

### Quaternions


For reasons of numerical stability, I'm going to use [quaternions](https://www.youtube.com/watch?v=3BR8tK-LuB0) to represent the camera's orientation/attitude in 3D space. 


```python
def create_rotation_matrix(Q0,Q1,Q2,Q3):
    R =[[Q0**2 + Q1**2 - Q2**2 - Q3**2, 2*(Q1*Q2 - Q0*Q3), 2*(Q0*Q2 + Q1*Q3)],
        [2*(Q1*Q2 + Q0*Q3), Q0**2 - Q1**2 + Q2**2 - Q3**2, 2*(Q2*Q3 - Q0*Q1)],
        [2*(Q1*Q3 - Q0*Q2), 2*(Q0*Q1 + Q2*Q3), (Q0**2 - Q1**2 - Q2**2 + Q3**2)]]
    return(R)

def normalize_quaternions(Q0,Q1,Q2,Q3):
    norm = pm.math.sqrt(Q0**2 + Q1**2 + Q2**2 + Q3**2)
    Q0 /= norm
    Q1 /= norm
    Q2 /= norm
    Q3 /= norm
    return(Q0,Q1,Q2,Q3)
```

![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/rotation_priors.png)



### Extrinsics



```python
def Rotate_Translate(X_est, Y_est, Z_est):
    Q1 = pm.StudentT('Xq', nu = 1.824, mu = 0.706, sigma = 0.015)
    Q2 = pm.StudentT('Yq', nu = 1.694, mu = -0.298, sigma = 0.004)
    Q3 = pm.StudentT('Zq', nu = 2.015, mu = 0.272, sigma = 0.011)
    Q0 = pm.StudentT('Wq', nu = 0.970, mu = 0.590, sigma = 0.019)
    
    Q0,Q1,Q2,Q3 = normalize_quaternions(Q0,Q1,Q2,Q3)
    
    R = create_rotation_matrix(Q0,Q1,Q2,Q3)
    
    # Define priors 
    X_translate = pm.Normal('X_translate', mu = -6.85, sigma = 10)
    Y_translate = pm.Normal('Y_translate', mu = -12.92, sigma = 10)
    Z_translate = pm.Normal('Z_translate', mu = 2.75, sigma = 5)
    
    RIC_0_3 = R[0][0] * -X_translate + R[0][1] * -Y_translate + R[0][2] * -Z_translate
    RIC_1_3 = R[1][0] * -X_translate + R[1][1] * -Y_translate + R[1][2] * -Z_translate
    RIC_2_3 = R[2][0] * -X_translate + R[2][1] * -Y_translate + R[2][2] * -Z_translate
    
    X_out = X_est * R[0][0] + Y_est * R[0][1] + Z_est * R[0][2] + RIC_0_3
    Y_out = X_est * R[1][0] + Y_est * R[1][1] + Z_est * R[1][2] + RIC_1_3
    Z_out = X_est * R[2][0] + Y_est * R[2][1] + Z_est * R[2][2] + RIC_2_3
    
    return(X_out, Y_out, Z_out)
```


### Intrinsics
```python
with pm.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
    X, Y, Z = Rotate_Translate(points3d[:,0], points3d[:,1], points3d[:,2])
    
    focal_length = pm.Normal('focal_length',mu = 2189.49, sigma = 11.74)
     
    k1 = pm.Normal('k1', mu = -0.327041, sigma = 0.5 * 0.327041)
    k2 = pm.Normal('k2', mu = 0.175031,  sigma = 0.5 * 0.175031)
    k3 = pm.Normal('k3', mu = -0.030751, sigma = 0.5 * 0.030751)
    
    c_x = pm.Normal('c_x', mu = 2268/2.0, sigma = 1000)
    c_y = pm.Normal('c_y', mu = 1503/2.0, sigma = 1000)
    
    px_est = X / Z
    py_est = Y / Z
    
    #Radial distortion
    r = pm.math.sqrt(px_est**2 + py_est**2)
    
    radial_distortion_factor = (1 + k1 * r + k2 * r**2 + k3 * r**3)
    px_est *= radial_distortion_factor
    py_est *= radial_distortion_factor
    
    px_est *= focal_length
    py_est *= focal_length

    px_est += c_x
    py_est += c_y
    
    error_scale = 5 #px
    
    delta = pm.math.sqrt((px - px_est)**2 + (py - py_est)**2)
    
    # Define likelihood
    likelihood = pm.Normal('rms_pixel_error', mu = delta, sigma = error_scale, observed=np.zeros(number_points))

    # Inference!
    trace = pm.sample(draws=10_000, init='adapt_diag', cores=3, tune=5_000)
```


## Results
```python
pm.plot_posterior(trace);
```

![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Posteriors.png)


```python
pm.summary(trace)
```


![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Posteriors2.png)


```python
pm.pairplot(trace, var_names=['X_translate','Y_translate','Z_translate'], divergences=True);
```

![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Posteriors3.png)

```python
pm.pairplot(trace, var_names=['k1', 'k2', 'k3'], divergences=True);
```

![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Posteriors4.png)

```python
pm.pairplot(trace, var_names=['c_x', 'c_y'], divergences=True);
```

![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Posteriors5.png)

```python
pm.pairplot(trace, var_names=['Wq', 'Xq','Yq','Zq'], divergences=True);
```

![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Posteriors6.png)


```python
sns.jointplot(trace[:]['X_translate'], trace[:]['Y_translate'], kind="hex");
```

![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Posteriors7.png)

```python
sns.jointplot(trace[:]['X_translate'], trace[:]['Z_translate'], kind="hex");
```

![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Posteriors8.png)

```python
sns.jointplot(trace[:]['c_x'], trace[:]['c_y'], kind="hex");
```

![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Posteriors9.png)

```python
sns.jointplot(trace[:]['Wq'], trace[:]['Xq'], kind="hex");
```


![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Posteriors10.png)








