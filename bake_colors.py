import bpy
import os
import random

root = 'C:\\Users\\user\\text2mesh\\dataset\\'

def bake_objs(obj_class: str):
    obj_folders = []
    for obj_id in os.listdir(root + f'annotated_models\\{obj_class}'):
        obj_folders.append(os.path.join(obj_id))

    outDir = root + f'baked_models\\{obj_class}'

    bpy.data.scenes["Scene"].render.engine = "CYCLES"
    bpy.data.scenes["Scene"].cycles.bake_type = "DIFFUSE"
    bpy.data.scenes["Scene"].render.bake.use_pass_direct = False
    bpy.data.scenes["Scene"].render.bake.use_pass_indirect = False
    bpy.data.scenes["Scene"].render.bake.use_pass_color = True
    bpy.data.scenes["Scene"].render.bake.target = "VERTEX_COLORS"

    for obj_folder in obj_folders:
        filename = os.path.join(root, 'annotated_models', obj_class, obj_folder, 'model.obj')
        target_file = os.path.join(outDir, obj_folder, "baked_model.ply")
        if not os.path.isdir(os.path.join(outDir, obj_folder)):
            os.mkdir(os.path.join(outDir, obj_folder))
        else:
            continue
        imported_object = bpy.ops.import_scene.obj(filepath=filename)
        obj_object = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = obj_object
        bpy.ops.object.join()
        
        print('Imported name: ', obj_object.name)

        mesh = obj_object.data

        if not mesh.vertex_colors:
            mesh.vertex_colors.new()
            
        bpy.ops.object.bake(type='DIFFUSE')
        
        bpy.ops.export_mesh.ply(filepath=target_file)
        
        bpy.ops.object.delete()


bake_objs('Table')
bake_objs('Chair')