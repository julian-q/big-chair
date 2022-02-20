import os
from nltk.corpus import wordnet as wn
import csv
import json
import shutil
from tqdm import tqdm

# This script is run by download_dataset.sh
captions_path = 'captions.tablechair.csv'
unannotated_models_path = 'unannotated_models/'
annotated_models_path = 'annotated_models/'

table_synset_id = str(wn.synset('table.n.01').offset()).zfill(8)
chair_synset_id = str(wn.synset('chair.n.01').offset()).zfill(8)

model2desc = {}
with open(captions_path, 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in tqdm(reader):
        category    = row['category']
        synset_id   = row['topLevelSynsetId']
        model_id    = row['modelId']
        description = row['description'].lower()

        if model_id in model2desc:
            model2desc[model_id].append(description)
        else:
            model2desc[model_id] = [description]

            model_src = os.path.join(unannotated_models_path, synset_id, model_id)
            model_dst = os.path.join(annotated_models_path, category, model_id)
            shutil.move(model_src, model_dst)

model2desc_path = os.path.join(annotated_models_path, 'annotations.json')
with open(model2desc_path, 'w') as model2desc_file:
    json.dump(model2desc, model2desc_file)
