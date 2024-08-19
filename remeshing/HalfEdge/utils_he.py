import numpy as np
from HalfEdge.half_edge import HalfEdgeTriMesh
from HalfEdge import utils_math
# compute std. deviation of edge length
def std_deviation_edge_len(he_trimesh:HalfEdgeTriMesh):
    edge_lengths = []
    for edge_index in he_trimesh:
            edge_lengths.append(he_trimesh.edge_len(edge_index))
    return np.std(edge_lengths)

def std_deviation_face_area(he_trimesh:HalfEdgeTriMesh):
    cross_product_norms = []
    for face_index, face_verts_inds in enumerate(he_trimesh.F):
        if face_index in he_trimesh.unreferenced_triangles:
            continue
        vertices = he_trimesh.V[face_verts_inds]
        cross_product_norms.append(utils_math.get_cross_product_norm(vertices))
    return np.std(cross_product_norms)
