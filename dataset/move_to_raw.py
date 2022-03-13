import os
import shutil

# run from dataset/
for mesh_class in ['Chair', 'Table']:
    for mesh_id in os.listdir(os.path.join('baked_models', mesh_class)):
        src = os.path.join('baked_models', mesh_class, mesh_id, 'baked_model.ply')
        dst = os.path.join('raw', mesh_id + '.ply')
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
