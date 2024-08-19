import os
import numpy as np
from HalfEdge.half_edge import HalfEdgeTriMesh
from HalfEdge import utils_math
from HalfEdge.utils_load import OUT_DIR


def std_deviation_edge_len(he_trimesh:HalfEdgeTriMesh):
    edge_lengths = []
    for edge_index in he_trimesh:
            edge_lengths.append(he_trimesh.edge_len(edge_index))
    return np.std(edge_lengths)

def face_area_stats(he_trimesh:HalfEdgeTriMesh):
    cross_product_norms = []
    for face_index, face_verts_inds in enumerate(he_trimesh.F):
        if face_index in he_trimesh.unreferenced_faces:
            continue
        vertices = he_trimesh.V[face_verts_inds]
        cross_product_norms.append(utils_math.get_cross_product_norm(vertices))
    face_area_std = np.std(cross_product_norms)
    # compute relative mean area error
    avg_prod_norms = np.mean(cross_product_norms)
    diff_vals = np.abs(cross_product_norms - avg_prod_norms)
    # find argmax
    avg_diff = np.mean(diff_vals)
    relative_mean_area_error = avg_diff / avg_prod_norms
    return face_area_std, relative_mean_area_error

def save_stats(he_trimesh, prefix, rewrite=True,extra=""):
    std_edge_len = std_deviation_edge_len(he_trimesh)
    std_area, relative_mean_area_error = face_area_stats(he_trimesh)
    stats_str = f"{prefix}\nstd. deviation edge len: {std_edge_len:.2f}, std. deviation face area: {std_area:.2f}, relative mean area error: {relative_mean_area_error:.2f}\n"
    if extra:
        stats_str = f"{extra}\n" + stats_str
    stats_path = os.path.join(OUT_DIR, he_trimesh.model_name + '_stats.txt')
    
    if rewrite:
        with open(stats_path, 'w') as stats_file:
            stats_file.write(stats_str)
    else:
        with open(stats_path, 'a') as stats_file:
            stats_file.write(stats_str)
    print(stats_str)
