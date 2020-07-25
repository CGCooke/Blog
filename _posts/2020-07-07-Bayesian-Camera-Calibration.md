---
toc: true
layout: post
description: Let's apply PyMC3 to our camera calibration problem
categories: [Bayesian, PyMC3, Computer Vision]
image: images/2020-07-07-Bayesian-Camera-Calibration/header.jpg
---

## The Context
In a previous [post](https://cgcooke.github.io/Blog/computer%20vision/optimisation/linear%20algebra/2020/02/23/An-Adventure-In-Camera-Calibration.html), I attempted to reverse-engineer information about a camera, from a photo it had taken of a scene. While I found a solution, I wasn't sure how confident I could be in the answer. I was also curious if I could improve the solution by injecting prior knowledge from other sources.


I had the idea to apply Bayesian analysis, and try to find a solution using *Makov Chain Monte Carlo* and [PyMC3](https://docs.pymc.io/). After a bit of searching, I also found this [paper](https://www.sciencedirect.com/science/article/pii/S0924271619302734), which told me that the idea wasn't completely outlandish.


In this post, we will combine a *prior* belief (probability distributions) about some of the camera's parameters, with measured 2D-3D scene correspondances. By combining these two sources of information, we can compute *posterior* distributions for each camera parameter. 

Because we have a probability distribution, we can understand how certain we are about each parameter. 

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


Ok, so now that we have loaded in our 2D and 3D point correspondences, we can now turn to representing the camera itself.

### Quaternions


For reasons of numerical stability, I'm going to use [quaternions](https://www.youtube.com/watch?v=3BR8tK-LuB0) to represent the camera's orientation/attitude in 3D space. 


```python
def create_rotation_matrix(Q0,Q1,Q2,Q3):
    #Create a rotation matrix from a Quaternion representation of an angle.
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


In a [previous post](https://cgcooke.github.io/Blog/computer%20vision/linear%20algebra/monte%20carlo%20simulation/2020/07/06/Vanishing-Points-In-Practice.html), I used sets of parallel lines in the image, to find the locations of vanishing points.

By using these [vanishing points](https://cgcooke.github.io/Blog/computer%20vision/linear%20algebra/monte%20carlo%20simulation/2020/04/10/Finding-Vanishing-Points.html), I was able to determine both an estimate for the orientation of the camera, as well as of the *intrinsic parameters*.


Because we are using quaternions to represent the orientation of the camera, we have 4 different components (X,Y,Z,W). 
![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/rotation_priors.png)


A *prior* is a probability distribution on a parameter, and I'm using [Student's T](https://docs.pymc.io/api/distributions/continuous.html#pymc3.distributions.continuous.StudentT) to model this distribution.

### Extrinsics



```python
Q1 = pm.StudentT('Xq', nu = 1.824, mu = 0.706, sigma = 0.015)
Q2 = pm.StudentT('Yq', nu = 1.694, mu = -0.298, sigma = 0.004)
Q3 = pm.StudentT('Zq', nu = 2.015, mu = 0.272, sigma = 0.011)
Q0 = pm.StudentT('Wq', nu = 0.970, mu = 0.590, sigma = 0.019)
```

To form a prior estimate for the location of the camera, I'm taking the results we found in this [post](https://cgcooke.github.io/Blog/computer%20vision/optimisation/linear%20algebra/2020/02/23/An-Adventure-In-Camera-Calibration.html), I'm taking the solution I generated as an initial estimate. However, I'm going to be open minded, and model the position using a normal distribution, with a standard deviation of 10 meters.

As I mentioned at the end of [post](https://cgcooke.github.io/Blog/computer%20vision/optimisation/linear%20algebra/2020/02/23/An-Adventure-In-Camera-Calibration.html), While I found a solution for the location of the camera, It was much lower that what I would have guessed. 

I can imagine the camera being about 7-10 meters off the ground, so by using a broad prior, we are saying that this outcome wouldn't be that surprising, if it was supported by the evidence.

```python
# Define  translation priors 
X_translate = pm.Normal('X_translate', mu = -6.85, sigma = 10)
Y_translate = pm.Normal('Y_translate', mu = -12.92, sigma = 10)
Z_translate = pm.Normal('Z_translate', mu = 2.75, sigma = 10)
```


Now we have to Rotate and Translate the points, in 3D space, according to the attitude and the position of the camera in 3D space.

$$[R | t]$$

Where $t$ is:

$$t = −RC$$

In a previous [post](https://cgcooke.github.io/Blog/computer%20vision/optimisation/linear%20algebra/2020/02/23/An-Adventure-In-Camera-Calibration.html), I was able to elegantly use Numpy. 

```python
#Camera Center
C = camera_params[3:6].reshape(3,1)
IC = np.hstack([np.eye(3),-C])
RIC = np.matmul(R,IC)

#Make points Homogeneous
points = np.hstack([points,np.ones((points.shape[0],1))])

#Perform Rotation and Translation
#(n,k), (k,m) -> (n,m)
points_proj = np.matmul(points,RIC.T)
```

However, we are a little constrained with PyMC3, so I explicitly (and inelegantly) carry out this as follows.


```python
RIC_0_3 = R[0][0] * -X_translate + R[0][1] * -Y_translate + R[0][2] * -Z_translate
RIC_1_3 = R[1][0] * -X_translate + R[1][1] * -Y_translate + R[1][2] * -Z_translate
RIC_2_3 = R[2][0] * -X_translate + R[2][1] * -Y_translate + R[2][2] * -Z_translate

X_out = X_est * R[0][0] + Y_est * R[0][1] + Z_est * R[0][2] + RIC_0_3
Y_out = X_est * R[1][0] + Y_est * R[1][1] + Z_est * R[1][2] + RIC_1_3
Z_out = X_est * R[2][0] + Y_est * R[2][1] + Z_est * R[2][2] + RIC_2_3
```



Now let's put it all together, 



```python
def Rotate_Translate(X_est, Y_est, Z_est):
    #Define rotation priors
    Q1 = pm.StudentT('Xq', nu = 1.824, mu = 0.706, sigma = 0.015)
    Q2 = pm.StudentT('Yq', nu = 1.694, mu = -0.298, sigma = 0.004)
    Q3 = pm.StudentT('Zq', nu = 2.015, mu = 0.272, sigma = 0.011)
    Q0 = pm.StudentT('Wq', nu = 0.970, mu = 0.590, sigma = 0.019)
    
    Q0,Q1,Q2,Q3 = normalize_quaternions(Q0,Q1,Q2,Q3)
    
    R = create_rotation_matrix(Q0,Q1,Q2,Q3)
    
    # Define  translation priors 
    X_translate = pm.Normal('X_translate', mu = -6.85, sigma = 10)
    Y_translate = pm.Normal('Y_translate', mu = -12.92, sigma = 10)
    Z_translate = pm.Normal('Z_translate', mu = 2.75, sigma = 10)
    
    RIC_0_3 = R[0][0] * -X_translate + R[0][1] * -Y_translate + R[0][2] * -Z_translate
    RIC_1_3 = R[1][0] * -X_translate + R[1][1] * -Y_translate + R[1][2] * -Z_translate
    RIC_2_3 = R[2][0] * -X_translate + R[2][1] * -Y_translate + R[2][2] * -Z_translate
    
    X_out = X_est * R[0][0] + Y_est * R[0][1] + Z_est * R[0][2] + RIC_0_3
    Y_out = X_est * R[1][0] + Y_est * R[1][1] + Z_est * R[1][2] + RIC_1_3
    Z_out = X_est * R[2][0] + Y_est * R[2][1] + Z_est * R[2][2] + RIC_2_3
    
    return(X_out, Y_out, Z_out)
```


### Intrinsics

For the intrinsics, I'm using a mixture of priors from the results of our optimisation, as well as what was identified in [this post](https://cgcooke.github.io/Blog/computer%20vision/linear%20algebra/monte%20carlo%20simulation/2020/07/06/Vanishing-Points-In-Practice.html).


```python
focal_length = pm.Normal('focal_length',mu = 2189.49, sigma = 11.74)
     
k1 = pm.Normal('k1', mu = -0.327041, sigma = 0.5 * 0.327041)
k2 = pm.Normal('k2', mu = 0.175031,  sigma = 0.5 * 0.175031)
k3 = pm.Normal('k3', mu = -0.030751, sigma = 0.5 * 0.030751)

c_x = pm.Normal('c_x', mu = 2268/2.0, sigma = 1000)
c_y = pm.Normal('c_y', mu = 1503/2.0, sigma = 1000)
```


```python
with pm.Model() as model:
    X, Y, Z = Rotate_Translate(points3d[:,0], points3d[:,1], points3d[:,2])
    
    focal_length = pm.Normal('focal_length',mu = 2189.49, sigma = 11.74)
     
    k1 = pm.Normal('k1', mu = -0.327041, sigma = 0.5 * 0.327041)
    k2 = pm.Normal('k2', mu = 0.175031,  sigma = 0.5 * 0.175031)
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

```


Finally we can use *Markov Chain Monte Carlo* (MCMC) to find the *posterior* distribution of the parameters. 

```python
with pm.Model() as model:
    # Inference!
    trace = pm.sample(draws=10_000, init='adapt_diag', cores=4, tune=5_000)
```


## Results

Now that the MCMC sampling has finished, let's look at the results:

```python
pm.plot_posterior(trace);
```

![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Posterior1.png)


From this we can see two things. In the left hand column, we can see the distribution of potential values for each parameter. In the right hand column, we can see how the MCMC sampler moved through this space over time. 

```python
pm.summary(trace)
```


![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Summary.png)


A number of things stand out. 


Firstly, the Z_centroid of the camera is now 7.44 meters off the ground, with a very small standard deviation (8.5cm).


Secondly, the y coordinate of the inferred principle point (c_y) is much closer to what we expect. Previously we found that it was located at ±2650 pixels, well outside the image. Now we find that it's somewhere in the broad region of ±1000 pixels. 

All things considered, we now have a solution where we understand how confident we are in each parameter, and is a solution that is more reasonable, than what we found in this [post](https://cgcooke.github.io/Blog/computer%20vision/optimisation/linear%20algebra/2020/02/23/An-Adventure-In-Camera-Calibration.html).

If we found more data, we could further refine our estimates, by using the *posterior* results as new *priors*. This is exciting, as it gives us the framework to evolve and update both our results, and how confident we are, over time.


## Additional plots

```python
pm.pairplot(trace, var_names=['X_translate','Y_translate','Z_translate'], divergences=True);
```

![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Posterior2.png)

```python
pm.pairplot(trace, var_names=['k1', 'k2', 'k3'], divergences=True);
```

![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Posterior3.png)

```python
pm.pairplot(trace, var_names=['c_x', 'c_y'], divergences=True);
```

![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Posterior4.png)

```python
pm.pairplot(trace, var_names=['Wq', 'Xq','Yq','Zq'], divergences=True);
```

![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Posterior5.png)


```python
sns.jointplot(trace[:]['X_translate'], trace[:]['Y_translate'], kind="hex");
```

![_config.yml]({{ site.baseurl }}/images/2020-07-07-Bayesian-Camera-Calibration/Posterior6.png)



