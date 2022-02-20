from nltk.corpus import wordnet as wn
import csv
import json
import os
import shutil
from tqdm import tqdm

captions_path = 'dataset/captions.tablechair.csv'
shapenet_path = 'dataset/ShapeNetCore.v2/'

annotated_path = 'dataset/annotated_models/'
os.mkdir(annotated_path)

table_synset_id = str(wn.synset('table.n.01').offset()).zfill(8)
chair_synset_id = str(wn.synset('chair.n.01').offset()).zfill(8)
table_path = os.path.join(annotated_path, table_synset_id)
chair_path = os.path.join(annotated_path, chair_synset_id)
os.mkdir(table_path)
os.mkdir(chair_path)

model2desc = {}
with open(captions_path) as csv_file:
    reader = csv.DictReader(csv_file)
    for row in tqdm(reader):
        synset_id   = row['topLevelSynsetId']
        model_id    = row['modelId']
        description = row['description']
        print(model_id)
        
        model2desc[model_id] = description
        
        model_src = os.path.join(shapenet_path,  synset_id, model_id)
        model_dst = os.path.join(annotated_path, synset_id, model_id)
        
        if os.path.isdir(model_src):
            shutil.move(model_src, model_dst)

model2desc_path = os.path.join(annotated_path, 'annotations.json')
with open(model2desc_path, 'w') as model2desc_file:
    json.dump(model2desc, model2desc_file)

