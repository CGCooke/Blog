---
toc: true
layout: post
description: In this post, I want to dig a little deeper into materials, and importing meshes.
categories: [Computer Vision,Blender]
image: images/2020-10-16-Meshes-And-Materials-In-Blender/header.png
---

The script
-------------

`blender --background --python myscript.py`

Let's walk through what `myscript.py` 
could look like:

```python
bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink = True)
```

```python
bpy.ops.mesh.primitive_plane_add(size=1000,location=(0, 0, 0), scale=(1, 1, 1))
```

### Importing Mesh

```python
def create_bunny():
        bpy.ops.import_scene.obj(filepath=os.getcwd()+"/stanford_bunny.obj")
        ob = bpy.data.objects["stanford_bunny"]
        ob.scale = (10,10,10)
        ob.location = (1,0,-0.35)
        ob.name = 'Bunny'
        
        #Perform subdivision
        bevel_mod = ob.modifiers.new('Subsurf', 'SUBSURF')
        bevel_mod.render_levels = 3
```

### Materials

```python
def create_material(object_name,material_name):
        mat = bpy.data.materials.new(name=material_name)
        bpy.data.objects[object_name].active_material = mat
        
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        return(nodes)

def create_ground_plane_material(object_name,material_name):
        nodes = create_material(object_name,material_name)
        nodes["Principled BSDF"].inputs[0].default_value = (0.7,0.7,0.7,1)
        nodes["Principled BSDF"].inputs[5].default_value = 1
        nodes["Principled BSDF"].inputs[7].default_value = 0.1

create_ground_plane_material("Plane","Plane_material")
```


```python

def create_bunny_material(object_name,material_name):
        nodes = create_material(object_name,material_name)
        nodes["Principled BSDF"].inputs[0].default_value =  (0.603828, 1, 0.707399, 1)
        nodes["Principled BSDF"].inputs[7].default_value = 0.1

        nodes["Principled BSDF"].inputs[14].default_value = 1.5
        nodes["Principled BSDF"].inputs[15].default_value = 1
        nodes["Principled BSDF"].inputs[16].default_value = 0.75

create_bunny_material("Bunny","Bunny_Material")
```


### Lights, Camera, Render!

```python
def configure_light():
        bpy.data.objects["Light"].data.type = 'AREA'
        bpy.data.objects["Light"].scale = (10,10,1)
        bpy.data.objects["Light"].location = (0,0,6)
        bpy.data.objects["Light"].rotation_euler = (0,0,0)

def configure_camera():
        bpy.data.objects["Camera"].location = (0.7, -4, 3)
        bpy.data.objects["Camera"].rotation_euler = (np.radians(60),0,0)

def configure_render():
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.render.filepath = os.getcwd()+"/render.png"
        bpy.context.scene.render.resolution_x = 1600
        bpy.context.scene.render.resolution_y = 1200
        bpy.context.scene.cycles.samples = 2560


configure_light()
configure_camera()
configure_render()

bpy.ops.render.render(write_still=True)
```


The Results
-------------

![_config.yml]({{ site.baseurl }}/images/2020-10-16-Meshes-And-Materials-In-Blender/render.png)


