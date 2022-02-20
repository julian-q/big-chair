mkdir all_models
cd all_models
# Download tables
curl -O http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/04379243.zip
unzip 04379243.zip
rm 04379243.zip
# Download chairs
curl -O http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/03001627.zip
unzip 03001627.zip
rm 03001627.zip
cd ..