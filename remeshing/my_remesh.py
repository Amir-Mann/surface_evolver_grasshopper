from HalfEdge.half_edge import HalfEdgeTriMesh

def split_long_edges(he_trimesh: HalfEdgeTriMesh, threshold:int):
    for h_idx in he_trimesh: # iterates only once over each half edge
        if he_trimesh.is_he_boundary(h_idx):
            continue
        edge_length = he_trimesh.edge_len(h_idx)
        if edge_length > threshold:
            he_trimesh.edge_split(h_idx)

def collapse_short_edges(he_trimesh: HalfEdgeTriMesh, threshold:int):
    for h_idx in he_trimesh: # iterates only once over each half edge
        if he_trimesh.is_he_collapsible(h_idx):
            edge_length = he_trimesh.edge_len(h_idx)
            if edge_length < threshold:
                he_trimesh.edge_collapse(h_idx)

def remesh(he_trimesh: HalfEdgeTriMesh, l:int=0):
    if l == 0:
        l = he_trimesh.get_average_edge_length()
    print("Average edge length:", l)
    # go over referenced edges and if length is greater than l, split
    # split long edges
    
        
    # collpase short edges
    # make sure not collapsing edge that is boundary, or that one of its vertices is boundary
    # also
    # https://stackoverflow.com/questions/26871162/half-edge-collapse
    # also
    # https://stackoverflow.com/questions/27049163/mesh-simplification-edge-collapse-conditions
    for h_idx in he_trimesh:
        edge_length = he_trimesh.edge_len(h_idx)
        if edge_length < (4*l/5):
            he_trimesh.edge_collapse(h_idx)
    
    # flip edges
    # skip boundary edges, or edges like 2, where flipping will create 
    # https://computergraphics.stackexchange.com/questions/12297/half-edge-criterion-to-check-if-an-edge-flip-is-illegal
    
    # TODO think of setting twin of boundary to an actual halfedge

if __name__ == "__main__":
    he_trimesh = HalfEdgeTriMesh.from_model_path("hex_grid_uv_03_ccw.obj")
    he_trimesh.edge_collapse(5)
    he_trimesh.visualize(wireframe=True, v_labels=True, e_labels=True, f_labels=True)
    remesh(he_trimesimesh, 0)
    he_trimesh.visualize(wireframe=True, v_labels=True, e_labels=True, f_labels=True)
    print('finished')