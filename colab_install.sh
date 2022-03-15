pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
python3.9 -m pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
python3.9 -m pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
python3.9 -m pip install torch-geometric
python3.9 -m pip install trimesh
python3.9 -m pip install spacy==3.2.3
python3.9 -m spacy download en_core_web_sm
python3.9 -m pip install transformers