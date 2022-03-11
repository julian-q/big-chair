import os
import shutil
from tqdm import tqdm

for object_class in ['Chair', 'Table']:
    for model_id in tqdm(os.listdir(os.path.join('annotated_models', object_class))):
        if os.path.isdir(os.path.join('annotated_models', object_class, model_id)):
            src_path = os.path.join('annotated_models', object_class, model_id, 'model.obj')
            if os.path.exists(src_path):
                dst_path = os.path.join('raw', model_id + '.obj')
                shutil.move(src_path, dst_path)

