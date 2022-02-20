1. `cd` into this directory.
2. Run `sh download_dataset.sh` to download all table and chair models from ShapeNet.
3. Run `python retrieve_annoted_models.py` to create and fill a directory full of models with English descriptions. A JSON file will also be created, mapping the id of each model (its directory name) to its description.
