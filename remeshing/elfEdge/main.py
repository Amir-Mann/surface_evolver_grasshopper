import open3d as o3d
import numpy as np
from open3d.geometry import HalfEdge, TriangleMesh, HalfEdgeTriangleMesh # type: ignore
from half_edge import HalfEdgeModel

from utils_load import *
from utils_vis import *
from remesh import *
import pdb

# he_model = load_halfedge_model_from_json("hex_grid.json")
he_model = load_halfedge_model_from_obj("hex_grid_uv_03_ccw.obj")
# or19 = list(he_model.one_ring(19))
# or19 = list(he_model.one_ring(22))
he_model.edge_collapse(29)
# he_model.edge_split_boundary(15)
# he_model.edge_flip(45)
# remesh(he_model, 0)
vis_halfedge_model_old(he_model, wireframe=True, vert_labels=True, edge_labels=True, face_labels=True)
print('finished')
# vis_halfedge_model(he_model, wireframe=True, vert_labels=False, edge_labels=False, face_labels=False)

