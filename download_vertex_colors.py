import bpy
import os
import random
import pickle

root = '/Users/julianquevedo/code/text2mesh/dataset'

def bake_objs(obj_class: str):
    obj_folders = []
    for obj_id in os.listdir(os.path.join(root, 'annotated_models', obj_class)):
        obj_folders.append(os.path.join(obj_id))

#    outDir = root + f'baked_models\\{obj_class}'

    bpy.data.scenes["Scene"].render.engine = "CYCLES"
    bpy.data.scenes["Scene"].cycles.bake_type = "DIFFUSE"
    bpy.data.scenes["Scene"].render.bake.use_pass_direct = False
    bpy.data.scenes["Scene"].render.bake.use_pass_indirect = False
    bpy.data.scenes["Scene"].render.bake.use_pass_color = True
    bpy.data.scenes["Scene"].render.bake.target = "VERTEX_COLORS"

    for obj_folder in obj_folders:
        print('baking', obj_folder)
        filename = os.path.join(root, 'annotated_models', obj_class, obj_folder, 'model.obj')
        imported_object = bpy.ops.import_scene.obj(filepath=filename)
        obj_object = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = obj_object
        bpy.ops.object.join()
        
        print('Imported name: ', obj_object.name)

        mesh = obj_object.data

        if not mesh.vertex_colors:
            mesh.vertex_colors.new()
            
        bpy.ops.object.bake(type='DIFFUSE')
    
        #how many loops do we have ?
        loops = len(mesh.loops)
        verts = len(mesh.vertices)
       
        pos2col = {}
       
        #go through each vertex color layer
        for vcol in mesh.vertex_colors:
            # look into each loop's vertex ? (need to filter out double entries)
            visit = verts * [False]
            colors = {}
            
            for l in range(loops):
                v = mesh.loops[l].vertex_index
                c = vcol.data[l].color
                if not visit[v]:
                    colors[v] = c
                    visit[v] = True
                    
            sorted(colors)
            print("Vertex-Colors of Layer:", vcol.name)
            for v, c in colors.items():
                pos = tuple(mesh.vertices[v].co)
                col = [c[0], c[1], c[2]]
                pos2col[pos] = col
                
        with open(os.path.join(root, 'annotated_models', obj_class, obj_folder, 'vertex_colors.pickle'), 'wb') as pos2col_file:
            pickle.dump(pos2col, pos2col_file, protocol=pickle.HIGHEST_PROTOCOL)
        
#        saved = {}
#        with open(os.path.join(root, 'annotated_models', obj_class, obj_folder, vertex_colors.pickle)', 'rb') as pos2col_file:
#            saved = pickle.load(pos2col_file)
#            
#        print(list(saved.items())[:10])

#        target_file = os.path.join(outDir, obj_folder, "baked_model.ply")
#        if not os.path.isdir(os.path.join(outDir, obj_folder)):
#            os.mkdir(os.path.join(outDir, obj_folder))
#        bpy.ops.export_mesh.ply(filepath=target_file)
        
        bpy.ops.object.delete()


bake_objs('Chair')
bake_objs('Table')
