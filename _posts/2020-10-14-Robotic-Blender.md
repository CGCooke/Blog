---
toc: true
layout: post
description: Robotic Blender
categories: [Computer Vision,Blender]
image: images/2020-10-14-Robotic-Blender/header.png
---

```python
def create_material(object_name,material_name, rgba):
        mat = bpy.data.materials.new(name=material_name)
        bpy.data.objects[object_name].active_material = mat
        
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes["Principled BSDF"].inputs[0].default_value = rgba
        nodes["Principled BSDF"].inputs[5].default_value = 1
        nodes["Principled BSDF"].inputs[7].default_value = 0.1


def configure_camera():
        bpy.data.objects["Camera"].location[0] = 5
        bpy.data.objects["Camera"].location[1] = -5
        bpy.data.objects["Camera"].location[2] = 4


        bpy.data.objects["Camera"].rotation_quaternion[0] = 0.892399
        bpy.data.objects["Camera"].rotation_quaternion[1] = 0.369644
        bpy.data.objects["Camera"].rotation_quaternion[2] = 0.099046
        bpy.data.objects["Camera"].rotation_quaternion[3] = 0.239118



def configure_light():
        bpy.data.objects["Light"].data.type = 'AREA'
        bpy.data.objects["Light"].scale[0] = 10
        bpy.data.objects["Light"].scale[1] = 10
        
def configure_render():
        bpy.context.scene.render.engine = 'CYCLES'

        bpy.context.scene.render.filepath = "/Users/cooke_c/Documents/Quality Platform/Simple2.png"
        bpy.context.scene.render.resolution_x = 1920
        bpy.context.scene.render.resolution_y = 1080
        

bpy.ops.mesh.primitive_plane_add(size=1000, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))


cube_scale = 0.5
bpy.data.objects["Cube"].scale = (cube_scale,cube_scale,cube_scale)
bpy.data.objects["Cube"].location = (0,0,cube_scale)


create_material("Cube","Cube_material",(3/255.0, 223/255.0, 252/255.0,1))
create_material("Plane","Plane_material",(252/255.0, 3/255.0, 235/255.0,1))

configure_camera()
configure_light()
configure_render()


bpy.ops.render.render(write_still=True)
```



Creating a Kalman Filter in 7 easy steps:
===============

At this point, I think it's worthwhile considering how all of these matrices are related to each other.
*Tim Babb* of Bzarg, has a fantastic diagram, which sets out how information flows through all of the filters mentioned above. 
If you haven't already, I strongly recommend you read his [post](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)  on how Kalman filters work 
![_config.yml]({{ site.baseurl }}/images/2020-10-14-Robotic-Blender/render.png)



Further reading
===============
Control theory is a broad an intellectually stimulating area, with broad applications.  [Brian Douglas](https://www.youtube.com/user/ControlLectures) has an incredible YouTube channel which I strongly recommend. 