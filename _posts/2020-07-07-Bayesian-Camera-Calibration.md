---
toc: true
layout: post
description: Let's apply PyMC3 to our camera calibration problem
categories: [Bayesian, PyMC3, Computer Vision]
image: images/2020-07-07-Bayesian-Camera-Calibration/header.jpg
---

Paragraph Header
===============


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

def Rotate_Translate(X_est, Y_est, Z_est):
    Q1 = pm.StudentT('Xq', nu = 1.983, mu = 0.707, sigma = 0.016)
    Q2 = pm.StudentT('Yq', nu = 1.799, mu = -0.298, sigma = 0.004)
    Q3 = pm.StudentT('Zq', nu = 2.178, mu = 0.272, sigma = 0.012)
    Q0 = pm.StudentT('Wq', nu = 1.545, mu = 0.583, sigma = 0.013)
    
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




```python
with pm.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
    X, Y, Z = Rotate_Translate(points3d[:,0], points3d[:,1], points3d[:,2])
    
    focal_length = pm.Normal('focal_length',mu = 2191, sigma = 11.50)
    
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



```python
pm.plot_posterior(trace);
```

```python
pm.summary(trace)
```

```python
pm.pairplot(trace, var_names=['X_translate','Y_translate','Z_translate'], divergences=True);
```

```python
pm.pairplot(trace, var_names=['k1', 'k2', 'k3'], divergences=True);
```

```python
pm.pairplot(trace, var_names=['c_x', 'c_y'], divergences=True);
```

```python
pm.pairplot(trace, var_names=['Wq', 'Xq','Yq','Zq'], divergences=True);
```

```python
sns.jointplot(trace[:]['X_translate'], trace[:]['Y_translate'], kind="hex");
```

```python
sns.jointplot(trace[:]['X_translate'], trace[:]['Z_translate'], kind="hex");
```

```python
sns.jointplot(trace[:]['c_x'], trace[:]['c_y'], kind="hex");
```

```python
sns.jointplot(trace[:]['Wq'], trace[:]['Xq'], kind="hex");
```