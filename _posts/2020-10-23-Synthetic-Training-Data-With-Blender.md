---
toc: true
layout: post
description: Let's learn how we can create synthetic imagery, for training machine learning models.
categories: [Computer Vision,Blender]
image: images/2020-10-23-Synthetic-Training-Data-With-Blender/header.png
---

Hello World 
-------------

Let's walk through what `myscript.py` 
could look like:

```python
import os
import bpy
```

### Objects

When *Blender* loads, the default scene already contains a cube, called *Cube*. 
Let's adjust it's position and scale.

```python 
cube_scale = 0.5
bpy.data.objects["Cube"].scale = (cube_scale,cube_scale,cube_scale)
bpy.data.objects["Cube"].location = (0,0,cube_scale)
```

Now we can alos create a ground plane, and add that to the scene.
```python 
bpy.ops.mesh.primitive_plane_add(size=1000, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
```


Our next task is to create materials for the objects we have added into the scene. 

### Material

```python
def create_material(object_name,material_name, rgba):
        mat = bpy.data.materials.new(name=material_name)
        bpy.data.objects[object_name].active_material = mat
        
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes["Principled BSDF"].inputs[0].default_value = rgba
        nodes["Principled BSDF"].inputs[5].default_value = 1
        nodes["Principled BSDF"].inputs[7].default_value = 0.1

```

```python 
create_material("Cube","Cube_material",(3/255.0, 223/255.0, 252/255.0,1))
create_material("Plane","Plane_material",(252/255.0, 3/255.0, 235/255.0,1))
```


### Lights

```python
def configure_light():
        bpy.data.objects["Light"].data.type = 'AREA'
        bpy.data.objects["Light"].scale[0] = 10
        bpy.data.objects["Light"].scale[1] = 10

configure_light()
```

### Camera

Now let's configure the camera's position, and orientation/attitude (Using quaternions).

```python
def configure_camera():
        bpy.data.objects["Camera"].location = (5, -5, 4)
        bpy.data.objects["Camera"].rotation_mode = 'QUATERNION'
        bpy.data.objects["Camera"].rotation_quaternion = (0.892399, 0.369644, 0.099046, 0.239118_

configure_camera()
```


### Action! (Renderer)

Finally, let's configure the renderer. I've chosen to use *Cycles*, which is a physically based renderer/ray tracer.

```python 
def configure_render():
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.render.filepath = os.getcwd()+"/render.png"
        bpy.context.scene.render.resolution_x = 1920
        bpy.context.scene.render.resolution_y = 1080

configure_render()
```

And we can finish by rendering the image, and writing it out as *render.png*.

```python 
bpy.ops.render.render(write_still=True)
```


The Results
-------------

![_config.yml]({{ site.baseurl }}/images/2020-10-23-Synthetic-Training-Data-With-Blender/render.png)

![_config.yml]({{ site.baseurl }}/images/2020-10-23-Synthetic-Training-Data-With-Blender/Depth.png)

![_config.yml]({{ site.baseurl }}/images/2020-10-23-Synthetic-Training-Data-With-Blender/Index.png)

