import os
so_far = 0
total = len(os.listdir(os.path.join('dataset', 'annotated_models', 'Table')))
for obj_folder in os.listdir(os.path.join('dataset', 'annotated_models', 'Table')):
    if os.path.exists(os.path.join('dataset', 'annotated_models', 'Table', obj_folder, 'vertex_colors.pickle')):
        so_far += 1

print(so_far / total * 100)