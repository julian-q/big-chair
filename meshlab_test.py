from numpy import trim_zeros
import pymeshlab
import numpy as np
from pymeshlab import Mesh
import trimesh

trimesh_scene = trimesh.load("textured/model.obj")

i = 0
for submesh in trimesh_scene.geometry.values():
    submesh.visual = submesh.visual.to_color()

    # vertices = submesh.vertices
    # colors = np.delete(submesh.visual.vertex_colors, 3, 1)

    # features = np.concatenate([vertices, colors], axis=1)
    # print(features)

    print(submesh.visual.vertex_colors)
    print()

    


# get the vertex colors 
# get the vertex colors from faces
# go





# ms = pymeshlab.MeshSet()
# ms.load_new_mesh("model.obj")
# ms.set_current_mesh(0)

# # ms.set_texture_per_mesh("images/texture0.jpg")
# # ms.transfer_texture_to_color_per_vertex()

# # # ms.compute_color_transfer_face_to_vertex()

# mesh = ms.current_mesh()

# ms.save_current_mesh("test_saved.obj")

# # TODO:
# # edge lengths
# # vertex colors