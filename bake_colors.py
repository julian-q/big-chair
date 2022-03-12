import bpy
import os

def bake_objs(obj_class: str):
    filenames = []
    for obj_id in os.listdir(f'annotated_models/{obj_class}'):
        filenames.append(os.path.join(obj_id, 'model.obj'))

    outDir = f'baked_models/{obj_class}'

    bpy.data.scenes["Scene"].render.engine = "CYCLES"
    bpy.data.scenes["Scene"].cycles.bake_type = "DIFFUSE"
    bpy.data.scenes["Scene"].render.bake.use_pass_direct = False
    bpy.data.scenes["Scene"].render.bake.use_pass_indirect = False
    bpy.data.scenes["Scene"].render.bake.use_pass_color = True
    bpy.data.scenes["Scene"].render.bake.target = "VERTEX_COLORS"

    for filename in filenames:
        imported_object = bpy.ops.import_scene.obj(filepath=filename)
        obj_object = bpy.context.selected_objects[0]
        print('Imported name: ', obj_object.name)

        bpy.context.view_layer.objects.active = obj_object

        obj = bpy.context.selected_objects[0]
        mesh = obj.data

        if not mesh.vertex_colors:
            mesh.vertex_colors = mesh.vertex_colors.new()
            
        bpy.ops.object.bake(type='DIFFUSE')

        target_file = outDir + filename + ".ply"
        bpy.ops.export_mesh.ply(filepath=target_file)
    
bake_objs('Chair')
bake_objs('Table')