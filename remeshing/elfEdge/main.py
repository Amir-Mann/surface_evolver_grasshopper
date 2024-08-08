import open3d as o3d
import numpy as np
from open3d.geometry import HalfEdge, TriangleMesh, HalfEdgeTriangleMesh # type: ignore
from half_edge import HalfEdgeModel

from utils_load import *
from utils_vis import *
from remesh import *

# halfedge_model = load_halfedge_model_from_json("hex_grid.json")
halfedge_model = load_halfedge_model_from_obj("hex_grid_uv_03_ccw.obj")
halfedge_model.edge_collapse(29)
# halfedge_model.edge_split_boundary(15)
# halfedge_model.edge_flip(45)
# remesh(halfedge_model, 0)
vis_halfedge_model(halfedge_model, wireframe=True, vert_labels=True, edge_labels=True, face_labels=True)
print('finished')
# vis_halfedge_model(halfedge_model, wireframe=True, vert_labels=False, edge_labels=False, face_labels=False)

