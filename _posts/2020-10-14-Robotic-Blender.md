---
toc: true
layout: post
description: Robotic Blender
categories: [Computer Vision,Blender]
image: images/2020-10-14-Robotic-Blender/header.png
---

What is Blender?
-------------


Why would we want to automate Blender?
-------------


How can we automate Blender?
-------------


`blender --background --python myscript.py`


```python
import os
import bpy
```

```python 
bpy.ops.mesh.primitive_plane_add(size=1000, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
```

```python 
cube_scale = 0.5
bpy.data.objects["Cube"].scale = (cube_scale,cube_scale,cube_scale)
bpy.data.objects["Cube"].location = (0,0,cube_scale)
```


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


```python
def configure_camera():
        bpy.data.objects["Camera"].location[0] = 5
        bpy.data.objects["Camera"].location[1] = -5
        bpy.data.objects["Camera"].location[2] = 4


        bpy.data.objects["Camera"].rotation_quaternion[0] = 0.892399
        bpy.data.objects["Camera"].rotation_quaternion[1] = 0.369644
        bpy.data.objects["Camera"].rotation_quaternion[2] = 0.099046
        bpy.data.objects["Camera"].rotation_quaternion[3] = 0.239118

```

```python
def configure_light():
        bpy.data.objects["Light"].data.type = 'AREA'
        bpy.data.objects["Light"].scale[0] = 10
        bpy.data.objects["Light"].scale[1] = 10
```

```python 
def configure_render():
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.render.filepath = os.getcwd()+"/render.png"
        bpy.context.scene.render.resolution_x = 1920
        bpy.context.scene.render.resolution_y = 1080
```



```python 

configure_camera()
configure_light()
configure_render()

bpy.ops.render.render(write_still=True)
```


The Results
-------------

![_config.yml]({{ site.baseurl }}/images/2020-10-14-Robotic-Blender/render.png)


