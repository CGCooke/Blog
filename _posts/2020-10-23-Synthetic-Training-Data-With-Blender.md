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

### Materials

```python 
def create_dragon_material(material_name,rgba):
        mat = bpy.data.materials.new(name=material_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes

        #Base Color
        nodes["Principled BSDF"].inputs[0].default_value = rgba

        #Subsurface
        nodes["Principled BSDF"].inputs[1].default_value = 0.5

        #Subsurface Color
        nodes["Principled BSDF"].inputs[3].default_value = rgba
        
        #Clearcoat 
        nodes["Principled BSDF"].inputs[12].default_value = 0.5
        return(mat)


def create_floor_material(material_name,rgba):
        mat = bpy.data.materials.new(name=material_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes

        #Base Color
        nodes["Principled BSDF"].inputs[0].default_value = rgba

        #Clearcoat 
        nodes["Principled BSDF"].inputs[12].default_value = 0.5
        return(mat)
```

### Objects
```python 
def create_dragon(location, rotation, rgba, index):
        #Load the mesh
        bpy.ops.import_mesh.ply(filepath=os.getcwd()+"/dragon_vrip.ply")
        ob = bpy.context.active_object #Set active object to variable

        ob.scale = (10,10,10)
        ob.location = location
        ob.rotation_euler = rotation
        bpy.context.object.pass_index = index

        #Create and add material to the object
        mat = create_dragon_material('Dragon_'+str(index)+'_Material',rgba=rgba)
        ob.data.materials.append(mat)

def create_floor():
        bpy.ops.mesh.primitive_plane_add(size=1000, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(100, 100, 1))
        mat = create_floor_material(material_name='Floor', rgba =  (0.9, 0.9, 0.9, 0)) 
        activeObject = bpy.context.active_object #Set active object to variable
        activeObject.data.materials.append(mat)
```

### Light & Camera 
```python

def configure_light():
        bpy.data.objects["Light"].data.type = 'AREA'
        bpy.data.objects["Light"].scale[0] = 20
        bpy.data.objects["Light"].scale[1] = 20


def configure_camera():
        bpy.data.objects["Camera"].location = (0,-4.96579,2.45831)
        bpy.data.objects["Camera"].rotation_euler = (np.radians(75),0,0)
        
```

### Renderer
```python 
def configure_render():
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.render.filepath = os.getcwd()+"/Metadata"
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        bpy.context.scene.cycles.samples = 1

        #Configure renderer to record object index
        bpy.context.scene.view_layers["View Layer"].use_pass_object_index = True


        # switch on nodes and get reference
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links

        ## clear default nodes
        for node in tree.nodes:
            tree.nodes.remove(node)


        image_output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        image_output_node.label = "Image_Output"
        image_output_node.base_path = "Metadata/Image"
        image_output_node.location = 400,0

        depth_output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_output_node.label = "Depth_Output"
        depth_output_node.base_path = "Metadata/Depth"
        depth_output_node.location = 400,-100

        index_output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        index_output_node.label = "Index_Output"
        index_output_node.base_path = "Metadata/Index"
        index_output_node.location = 400,-200

        render_layers_node = tree.nodes.new(type="CompositorNodeRLayers")
        render_layers_node.location = 0,0

        links.new(render_layers_node.outputs[0], image_output_node.inputs[0])
        links.new(render_layers_node.outputs[2], depth_output_node.inputs[0])
        links.new(render_layers_node.outputs[3], index_output_node.inputs[0])
```

### Execution
```python 
bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink = True)


create_floor()
create_dragon(location=(0,0.78,-0.56), rotation=(np.radians(90),0,0), rgba=(0.799, 0.125, 0.0423, 1), index=1)
create_dragon(location=(-1.5,4.12,-0.56), rotation=(np.radians(90),0,np.radians(227)), rgba=(0.0252, 0.376, 0.799, 1), index=2)
create_dragon(location=(1.04,2.7,-0.56), rotation=(np.radians(90),0,np.radians(129)), rgba=(0.133, 0.539, 0.292, 1), index=3)

configure_camera()
configure_light()
configure_render()
bpy.ops.render.render(write_still=True)
```


The Results
-------------

![_config.yml]({{ site.baseurl }}/images/2020-10-23-Synthetic-Training-Data-With-Blender/render.png)

![_config.yml]({{ site.baseurl }}/images/2020-10-23-Synthetic-Training-Data-With-Blender/Depth.png)

![_config.yml]({{ site.baseurl }}/images/2020-10-23-Synthetic-Training-Data-With-Blender/Index.png)

