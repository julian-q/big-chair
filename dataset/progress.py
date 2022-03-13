import os
import shutil


for obj_class in ['Chair', 'Table']:
    so_far = 0
    for obj_folder in os.listdir(os.path.join('dataset', 'annotated_models', obj_class)):
        src = os.path.join('dataset', 'annotated_models', obj_class, obj_folder)
        dst = os.path.join('dataset', 'colored_models', obj_class, obj_folder)
        if os.path.exists(os.path.join(src, 'vertex_colors.pickle')):
            so_far += 1
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
    print(obj_class, 'so far:', so_far)
