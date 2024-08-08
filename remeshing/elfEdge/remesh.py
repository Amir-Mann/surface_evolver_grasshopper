import numpy as np
import open3d as o3d

from half_edge import HalfEdgeModel
from utils_vis import vis_halfedge_model

def remesh(he_model: HalfEdgeModel, l=0):
    if l == 0:
        l = he_model.average_edge_length()
    print("Average edge length:", l)
    # go over referenced edges and if length is greater than l, split
    
    seen_edges = set()
    # split long edges
    for h_idx in range(len(he_model.half_edges)):
        if h_idx in seen_edges or h_idx in he_model.unreferenced_half_edges:
            continue
        edge_length = he_model.edge_len(h_idx)
        if edge_length > (5*l/4):
            if he_model.get_twin_index(h_idx)==-1:
                he_model.edge_split_boundary(h_idx)    
            else:
                he_model.edge_split(h_idx)
        seen_edges.add(h_idx)
        seen_edges.add(he_model.get_twin_index(h_idx))
    # collpase short edges
    # make sure not splitting edge that is boundary, or that one of its vertices is boundary
    # also
    # https://stackoverflow.com/questions/27049163/mesh-simplification-edge-collapse-conditions
    
    # flip edges
    # skip boundary edges, or edges like 2, where flipping will create 
    # https://computergraphics.stackexchange.com/questions/12297/half-edge-criterion-to-check-if-an-edge-flip-is-illegal
    
    # TODO think of setting twin of boundary to an actual halfedge
