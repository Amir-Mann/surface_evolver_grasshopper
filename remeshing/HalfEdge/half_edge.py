from typing import List, Union
import numpy as np
import numpy.typing as npt
from HalfEdge.utils_load import get_np_V_F
from HalfEdge.utils_vis import vis_he_trimesh
from HalfEdge.utils_math import get_normal, get_compactness, get_area
class HalfEdge:
    def __init__(self, 
                 idx: int  =-1,
                 next: int =-1,
                 twin: int =-1,
                 face: int =-1,
                 vertex_indices: List[int] = [-1,-1],
                 source_bdry: bool = False,
                 target_bdry: bool = False): 
        self.idx = idx
        self.twin = twin
        self.next = next
        self.face = face
        self.source_bdry = source_bdry
        self.target_bdry = target_bdry
        # if vertex_indices are tuple, turn to list
        if isinstance(vertex_indices, tuple):
            vertex_indices = list(vertex_indices)
        self.vertex_indices = vertex_indices
    def __repr__(self) -> str:
        source_bdry_str = 'B' if self.source_bdry else 'I' # boundary or interior
        target_bdry_str = 'B' if self.target_bdry else 'I'
        return f"HE{self.idx}({source_bdry_str}{self.vertex_indices}{target_bdry_str}),twin={self.twin},next={self.next},face={self.face}"
        
class HalfEdgeTriMesh:
    ############ SETUP ############
    @classmethod
    def from_model_path(cls, model_path: str):
        V,F = get_np_V_F(model_path)
        return cls(V,F)
    
    def __init__(self,
                 V : npt.NDArray[np.float64], # shape (#vertices,3)
                 F : npt.NDArray[np.int32]): # shape (#faces,3)
        half_edges_d = dict()
        self.half_edges = []
        self.V = V # np (#verts,3) float64
        self.F = F # np (#faces,3) int32
        self.n_vertices = len(V)
        self.unreferenced_vertices = []
        self.unreferenced_faces = []
        self.unreferenced_half_edges = []
        
        for face_idx, verts in enumerate(F):
            vi_idx = verts[0]
            vj_idx = verts[1]
            vk_idx = verts[2]
            eij = (vi_idx, vj_idx)
            ejk = (vj_idx, vk_idx)
            eki = (vk_idx, vi_idx)
            ij_idx = len(self.half_edges)
            jk_idx = ij_idx+1
            ki_idx = ij_idx+2
            he_ij = HalfEdge(idx=ij_idx, next=jk_idx, face=face_idx, vertex_indices=eij)
            he_jk = HalfEdge(idx=jk_idx, next=ki_idx, face=face_idx, vertex_indices=ejk)
            he_ki = HalfEdge(idx=ki_idx, next=ij_idx, face=face_idx, vertex_indices=eki)
            self.half_edges.extend([he_ij, he_jk, he_ki])
            half_edges_d[eij] = he_ij
            half_edges_d[ejk] = he_jk
            half_edges_d[eki] = he_ki
        
        # populate the twins amd boundary
        boundary_edges_d = dict() # mapping source vertex to he
        for edge, he in list(half_edges_d.items()):
            twin_edge = (edge[1], edge[0])
            twin_he = half_edges_d.get(twin_edge, None)
            if twin_he:
                he.twin = twin_he.idx
                he.source_bdry = False; he.target_bdry = False
            else: # set boundary edge and create its twin, also a boundary half-edge
                twin_idx = len(self.half_edges)
                he.twin = twin_idx
                he.source_bdry = True; he.target_bdry = True
                # create twin 
                twin_he = HalfEdge(idx=twin_idx, twin=he.idx, face=-1, vertex_indices=twin_edge, source_bdry=True, target_bdry=True)
                boundary_edges_d[edge[1]] = twin_he
                self.half_edges.append(twin_he)
        # populate boundary edges' nexts.
        for he in list(boundary_edges_d.values()):
            next_he = boundary_edges_d[he.vertex_indices[1]] # if not - error in manifold
            he.next = next_he.idx
            
        # go over one ring of each boundary edge, to set vertices as boundary
        for he in list(boundary_edges_d.values()):
            one_ring = self.one_ring(he.idx)
            for h_idx in one_ring:
                he_ring = self.half_edges[h_idx]
                he_ring.source_bdry = True
                self.half_edges[he_ring.twin].target_bdry = True

    def __iter__(self):
        # iterate over referenced(!) edges(!) not half edges
        skip = set()
        E = len(self.half_edges)
        for h_index in range(E):
            if h_index in self.unreferenced_half_edges:
                continue
            twin_index = self.half_edges[h_index].twin
            if twin_index in skip:
                continue
            skip.add(twin_index)
            
            yield h_index
    
    def visualize(self, wireframe:bool = True, v_labels:bool=True, e_labels:bool=True, f_labels:bool=True):
        vis_he_trimesh(self, wireframe=wireframe, v_labels=v_labels, e_labels=e_labels, f_labels=f_labels)
    
    ############ Half Edges ############
            
    def create_half_edge(self, next:int=-1, face:int=-1, twin:int=-1, vertex_indices:list=[-1, -1], source_bdry:bool=False, target_bdry:bool=False):
        he_idx = len(self.half_edges)
        h = HalfEdge(idx=he_idx, next=next, face=face, twin=twin, vertex_indices=vertex_indices, source_bdry=source_bdry, target_bdry=target_bdry)
        self.half_edges.append(h)
        return h
    
    def update_half_edge(self, h_index:int, next = None, face = None, twin = None, vertex_indices = None, source_bdry = None, target_bdry = None):
        he = self.half_edges[h_index]
        if next is not None: he.next = next
        if face is not None: he.face = face
        if twin is not None: he.twin = twin
        if vertex_indices is not None: he.vertex_indices = vertex_indices
        if source_bdry is not None: he.source_bdry = source_bdry
        if target_bdry is not None: he.target_bdry = target_bdry
        
    ############ GETTERS ############
    def get_vertices_by_indices(self, indices):
        # return (n,3) float64 for n=len(indices)
        return self.V[indices]
    
    def is_he_boundary(self, h_index):
        twin_index = self.half_edges[h_index].twin
        return self.half_edges[h_index].face==-1 or self.half_edges[twin_index].face==-1
    
    def is_he_collapsible(self, h_index):
        # an halfedge isn't collapsible if it's boundary, or both of its vertices are boundary
        he = self.half_edges[h_index]
        if (he.source_bdry and he.target_bdry):
            return False
        # check for connectivity, as in https://stackoverflow.com/a/27049418/4399305
        # denote he source vertex as v0 and target vertex as v1
        v0_ring = self.one_ring(h_index) # all he that have v0 as source
        v1_ring = self.one_ring(he.twin) # all he that have v1 as source
        
        v0_ring_target_vertices = {self.half_edges[he_idx].vertex_indices[1] for he_idx in v0_ring}
        v1_ring_target_vertices = {self.half_edges[he_idx].vertex_indices[1] for he_idx in v1_ring}
        # check intersection
        if len(v0_ring_target_vertices.intersection(v1_ring_target_vertices)) > 2:
            return False
        
        # check for geometry, as in https://stackoverflow.com/a/27049418/4399305

    def get_face_midpoint(self, t_idx):
        indices = self.F[t_idx]
        vertices_t = self.get_vertices_by_indices(indices)
        return np.mean(vertices_t, axis=0)
    
    def get_end_vertex_index(self, h_index:int):
        return self.half_edges[h_index].vertex_indices[1]
    
    def get_start_vertex_by_edge(self, h_index:int):
        v0_index = self.half_edges[h_index].vertex_indices[0]
        return self.get_vertices_by_indices(v0_index)
    
    def get_end_vertex_by_edge(self, h_index:int):
        v1_index = self.get_end_vertex_index(h_index)
        return self.get_vertices_by_indices(v1_index)
    
    def get_vertices_by_edge(self, h_index):
        indices = self.half_edges[h_index].vertex_indices
        return self.get_vertices_by_indices(indices)
    
    def edge_len(self, h_index):
        vertices = self.get_vertices_by_edge(h_index)
        return np.linalg.norm(vertices[0]-vertices[1])
    
    def num_half_edges(self, clean:bool = False):
        return len(self.half_edges) - len(self.unreferenced_half_edges) if clean else len(self.half_edges)
    
    def num_vertices(self, clean:bool = False):
        return len(self.V) - len(self.unreferenced_vertices) if clean else len(self.V)
    
    def num_faces(self, clean:bool = False):
        return len(self.F) - len(self.unreferenced_faces) if clean else len(self.F)
    
    def get_edge_midpoint(self, h_index):
        vertices = self.get_vertices_by_edge(h_index)
        return np.mean(vertices, axis=0)
    
    def get_vertices(self):
        # get referenced vertices
        referenced_vertices_d= dict() # map int to (3,) float64
        for v_idx in range(0, len(self.V)):
            if v_idx in self.unreferenced_vertices:
                continue
            referenced_vertices_d[v_idx] = self.V[v_idx]
        return referenced_vertices_d
    
    def get_edges_midpoints(self):
        E = len(self.half_edges)
        seen_edges = set()
        midpoints_d = dict()
        for h_index in range(E):
            if h_index in self.unreferenced_half_edges or h_index in seen_edges:
                continue
            
            midpoints_d[h_index] = self.get_edge_midpoint(h_index)
            seen_edges.add(h_index)
            seen_edges.add(self.half_edges[h_index].twin)
        return midpoints_d
    
    def get_triangles_midpoints(self):
        F = len(self.F)
        midpoints_d = dict()
        for t_idx in range(F):
            if t_idx in self.unreferenced_faces:
                continue
            midpoints_d[t_idx] = self.get_face_midpoint(t_idx)
        return midpoints_d
    
    def get_average_edge_length(self):
        total_length = 0
        num_edges = 0 # not half edges
        for edge_index in self:
            total_length += self.edge_len(edge_index)
            num_edges += 1
        return total_length/num_edges
    
    def get_triangle_indices_by_edge(self, h_index:int): 
        # returns (3,) int32 of of the triangle indices that the half-edge belongs to
        f_index = self.half_edges[h_index].face
        return self.F[f_index]
    
    def get_triangle_vertices_by_edge(self, h_index:int):
        # returns (3,3) float64 of the triangle vertices that the half-edge belongs to
        t_verts_indices = self.get_triangle_indices_by_edge(h_index) # (3,) int32
        # sort indices so first one will be like h_index source vertex
        source_vertex_index = self.half_edges[h_index].vertex_indices[0]
        roll_idx = np.argwhere(t_verts_indices==source_vertex_index)
        if roll_idx:
            t_verts_indices = np.roll(t_verts_indices, -roll_idx[0][0], axis=0)
        t_verts_positions = self.V[t_verts_indices] # (3,3) float64
        return t_verts_positions
    
    def valence(self, h_index:int):
        # return number of edges that leave source vertex
        # todo
        h = self.half_edges[h_index]
        v0_one_ring = list(self.one_ring(h_index))
        return len(v0_one_ring)
    
    def adjacent_half_edges(self, h_index:int):
        nh_index = self.half_edges[h_index].next # h1
        nnh_index = self.half_edges[nh_index].next # h2
        th_index = self.half_edges[h_index].next # h3
        nth_index = self.half_edges[th_index].next # h4
        nnth_index = self.half_edges[nth_index].next # h5
        return nh_index, nnh_index, nth_index, nnth_index # h1, h2, h4, h5
    
    def adjacent_triangles(self, h_index: int):
        f0_index = self.half_edges[h_index].face
        th_index = self.half_edges[h_index].twin
        f1_index = self.half_edges[th_index].face
        return f0_index, f1_index
    
    # set vertices and triangles methods
    def replace_triangle_by_index(self, t_index: int, new_t_verts_indices: np.ndarray): # new_t_verts_indices is (3,) int32
        self.F[t_index] = new_t_verts_indices
    
    def update_triangle_by_vertex_indices(self, h_index:int, v_index_new:int, v_index_old:int):
        t_index = self.half_edges[h_index].face
        t_verts_indices = self.F[t_index] #(3,) int32
        t_verts_indices[t_verts_indices==v_index_old] = v_index_new
        self.replace_triangle_by_index(t_index, t_verts_indices)
    
    def normal(self, h_index: int, normalize:bool=True) :
        return get_normal(self.get_triangle_vertices_by_edge(h_index),normalize=normalize)
    
    ############ RING METHODS ############
    def get_next_index_on_ring(self, h_index:int, clockwise:bool):
        if clockwise:
            th_index = self.half_edges[h_index].twin
            return self.half_edges[th_index].next
        else:
            nh_index = self.half_edges[h_index].next
            nnh_index = self.half_edges[nh_index].next
            return self.half_edges[nnh_index].twin
    
    ### INDEX RING METHODS ###
    def one_ring(self, h_index:int, clockwise:bool=True):
        # return half-edges whose source-vertex is like h_index's. returns in clockwise manner
        last = h_index
        while True:
            nh_index = self.get_next_index_on_ring(h_index, clockwise)
            yield nh_index
            if nh_index == last:
                break
            h_index = nh_index 
        
    def vertex_index_ring(self, h_index:int):
        # for h_index one ring, return list of the half-edges' target vertices indices
        # i.e. a list of ints
        # assume h_index is a half-edge with source vertex v0
        v0_one_ring = self.one_ring(h_index)
        return [self.half_edges[he_idx].vertex_indices[1] for he_idx in v0_one_ring]
    
    ### POSITION RING METHODS ###
    def edge_ring(self, h_index:int):
        # for h_index one_ring edges, returns a list of their vertices (instead of indices)
        # i.e. a list of  np.array (2,3) float64
        v0_one_ring = self.one_ring(h_index) # assume h_index is a half-edge with source vertex v0
        return [self.get_vertices_by_edge(h_index) for h_index in v0_one_ring]
    
    def vertex_ring(self, h_index:int):
        # for h_index edge ring, returns a list of the target vertices positions,
        # i.e. a list of (3,) float64
        v0_edge_ring = self.edge_ring(h_index) # list of (2,3) float64, assume h_index is a half-edge with source vertex v0
        return [edge[1] for edge in v0_edge_ring]
    
    def triangle_ring(self, h_index:int):
        # for h_index one_ring edges, returns their triangles (instead of indices)
        # i.e. a list of  np.array (3,3) float64
        v0_one_ring = self.one_ring(h_index) # assume h_index is a half-edge with source vertex v0
        return [self.get_triangle_vertices_by_edge(h_index) for h_index in v0_one_ring]
    
    def normal_ring(self, h_index:int, normalize:bool=True):
        # for h_index triangle_ring, returns their triangles normals
        # i.e. a list of (3,) float64
        v0_triangle_ring = self.triangle_ring(h_index) # list of (3,3) float64, assume h_index is a half-edge with source vertex v0
        return [get_normal(triangle, normalize=normalize) for triangle in v0_triangle_ring]
    
    def gravity_ring(self, h_index:int):
        vertex_ring = [] # list of (n,3)
        weighted_normal_ring = [] # list (n,3), each normal of triangle weighted by twice triangle area
        v0_one_ring = self.one_ring(h_index) 
        for h_idx in v0_one_ring:
            edge = self.get_vertices_by_edge(h_idx)
            vertex_ring.append(edge[1])
            weighted_normal_ring.append(self.normal(h_idx, normalize=False))
            
        return vertex_ring, weighted_normal_ring
        
    
    def compactness_ring(self, h_index:int):
        # for h_index triangle ring, how much each triangle is close to an equilateral triangle
        # i.e. a list of float64
        v0_triangle_ring = self.triangle_ring(h_index)
        return [get_compactness(triangle) for triangle in v0_triangle_ring]

    ############ half-edge modifying operations ############
    
    def edge_split(self, h0_index:int):
        h0_twin_index = self.half_edges[h0_index].twin
        
        is_h0_bdry = self.half_edges[h0_index].face == -1 or self.half_edges[h0_twin_index].face == -1
        if is_h0_bdry:
            return self.edge_split_boundary(h0_index)
        # assert not is_h0_bdry, f"edge_split({h0_index}) called but is boundary" # TODO
            

        E = self.num_half_edges()
        T = self.num_faces()
        V = self.num_vertices()

        # half edges indices
        h1_index = self.half_edges[h0_index].next
        h2_index = self.half_edges[h1_index].next
        h3_index = self.half_edges[h0_index].twin

        h4_index = self.half_edges[h3_index].next
        h5_index = self.half_edges[h4_index].next
        # triangles indices
        t0_index = self.half_edges[h0_index].face
        t1_index = self.half_edges[h3_index].face

        # create new half edges indices
        h6_index = E+0
        h7_index = E+1
        h8_index = E+2
        h9_index = E+3
        h10_index = E+4
        h11_index = E+5
        h12_index = E+6
        h13_index = E+7

        # create new triangles indices
        t2_index = T+0
        t3_index = T+1
        t4_index = T+2
        t5_index = T+3

        # create new vertex
        v0_index, v1_index = self.half_edges[h4_index].vertex_indices
        v2_index, v3_index = self.half_edges[h1_index].vertex_indices
        v4_index = V+0
        # create new vertex
        vertices = self.get_vertices_by_indices([v0_index, v2_index])
        v4 = vertices.mean(axis=0)
        
        # create new half edges
        h6 = self.create_half_edge(
            next=h7_index,
            face=t2_index,
            twin=h8_index,
            vertex_indices=[v0_index, v4_index],
            source_bdry = self.half_edges[h0_index].source_bdry,
            target_bdry = False
        )

        h7 = self.create_half_edge(
            next=h2_index,
            face=t2_index,
            twin=h12_index,
            vertex_indices=[v4_index, v3_index],
            source_bdry = False, # actually =is_h0_bdry, but then will call edge_split_boundary
            target_bdry = self.half_edges[h1_index].target_bdry
        )

        h8 = self.create_half_edge(
            next=h4_index,
            face=t3_index,
            twin=h6_index,
            vertex_indices=[v4_index, v0_index],
            source_bdry = False, # actually =is_h0_bdry, but then will call edge_split_boundary
            target_bdry = self.half_edges[h0_index].source_bdry
        )

        h9 = self.create_half_edge(
            next=h8_index,
            face=t3_index,
            twin=h10_index,
            vertex_indices=[v1_index, v4_index],
            source_bdry = self.half_edges[h4_index].target_bdry,
            target_bdry = False, # actually =is_h0_bdry, but then will call edge_split_boundary
        )

        h10 = self.create_half_edge(
            next=h5_index,
            face=t4_index,
            twin=h9_index,
            vertex_indices=[v4_index, v1_index],
            source_bdry = False, # actually =is_h0_bdry, but then will call edge_split_boundary
            target_bdry = self.half_edges[h4_index].target_bdry,
        )

        h11 = self.create_half_edge(
            next=h10_index,
            face=t4_index,
            twin=h13_index,
            vertex_indices=[v2_index, v4_index],
            source_bdry = self.half_edges[h3_index].source_bdry,
            target_bdry = False, # actually =is_h0_bdry, but then will call edge_split_boundary
        )

        h12 = self.create_half_edge(
            next=h13_index,
            face=t5_index,
            twin=h7_index,
            vertex_indices=[v3_index, v4_index],
            source_bdry = self.half_edges[h1_index].target_bdry,
            target_bdry = False, # actually =is_h0_bdry, but then will call edge_split_boundary
        )

        h13 = self.create_half_edge(
            next=h1_index,
            face=t5_index,
            twin=h11_index,
            vertex_indices=[v4_index, v2_index],
            source_bdry = False, # actually =is_h0_bdry, but then will call edge_split_boundary
            target_bdry = self.half_edges[h0_index].target_bdry,
        )


        # create new triangles
        t2 = np.array([v4_index, v3_index, v0_index])
        t3 = np.array([v4_index, v0_index, v1_index])
        t4 = np.array([v4_index, v1_index, v2_index])
        t5 = np.array([v4_index, v2_index, v3_index])

        

        # update half edges
        self.update_half_edge(
            h1_index,
            next=h12_index,
            face=t5_index
        )

        self.update_half_edge(
            h2_index,
            next=h6_index,
            face=t2_index
        )

        self.update_half_edge(
            h4_index,
            next=h9_index,
            face=t3_index
        )

        self.update_half_edge(
            h5_index,
            next=h11_index,
            face=t4_index
        )

        # insert new half edges
        # self.half_edges += [h6, h7, h8, h9, h10, h11, h12, h13] # caution: extend() does not work.

        # insert new triangles
        self.F = np.vstack([self.F, t2, t3, t4, t5])

        # insert new vertex
        self.V = np.vstack([self.V, v4])

        # save unreferenced half edges
        self.unreferenced_half_edges.extend([h0_index, h3_index])

        # save unreferenced triangles
        self.unreferenced_faces.extend([t0_index, t1_index])
      
    def edge_split_boundary(self, h0_index:int):
        h0 = self.half_edges[h0_index]
        h3 = self.half_edges[h0.twin]
        is_h0_bdry = h0.face == -1 or h3.face == -1
        assert is_h0_bdry, f"edge_split_boundary({h0_index}) called but isn't boundary"
        if h0.face==-1: # prefer to run over inside half-edge than boundary
            h0_index = h0.twin 
            h0, h3 = h3, h0
        E = self.num_half_edges()
        T = self.num_faces()
        V = self.num_vertices()
        
        # get data
        h1_index = self.half_edges[h0_index].next
        h2_index = self.half_edges[h1_index].next
        h3_index = h3.idx
        
        t0_index = h0.face
        
        v0_index, v2_index = self.half_edges[h0_index].vertex_indices
        _, v3_index = self.half_edges[h1_index].vertex_indices
        
        # create new indices
        h6_index = E+0
        h7_index = E+1
        h8_index = E+2
        h11_index = E+3
        h12_index = E+4
        h13_index = E+5
        
        t2_index = T+0
        t5_index = T+1
        
        v4_index = V+0
        
        # create new half edges
        
        h6 = self.create_half_edge(
            next=h7_index,
            face=t2_index,
            twin=h8_index,
            vertex_indices=[v0_index,v4_index],
            source_bdry = True,
            target_bdry = True
        )
        h7 = self.create_half_edge(
            next=h2_index,
            face=t2_index,
            twin=h12_index,
            vertex_indices=[v4_index,v3_index],
            source_bdry = True,
            target_bdry = self.half_edges[h1_index].target_bdry
        )
        h8 = self.create_half_edge(
            next=h3.next,
            face=-1,
            twin=h6_index,
            vertex_indices=[v4_index,v0_index],
            source_bdry = True,
            target_bdry = self.half_edges[h2_index].target_bdry
        )
        h11 = self.create_half_edge(
            next=h8_index,
            face=-1,
            twin=h13_index,
            vertex_indices=[v2_index,v4_index],
            source_bdry = True,
            target_bdry = True
        )
        h12 = self.create_half_edge(
            next=h13_index,
            face=t5_index,
            twin=h7_index,
            vertex_indices=[v3_index,v4_index],
            source_bdry = self.half_edges[h1_index].target_bdry,
            target_bdry = True
        )
        h13 = self.create_half_edge(
            next=h1_index,
            face=t5_index,
            twin=h11_index,
            vertex_indices=[v4_index,v2_index],
            source_bdry=True,
            target_bdry=True
        )
        
        # create new triangles
        t2 = np.array([v4_index, v3_index, v0_index])
        t5 = np.array([v4_index, v2_index, v3_index])
        
        # create new vertex
        vertices = self.get_vertices_by_indices([v0_index, v2_index])
        v4 = vertices.mean(axis=0)
        
        # update half edges
        self.update_half_edge(
            h1_index,
            next=h12_index,
            face=t5_index
        )
        self.update_half_edge(
            h2_index,
            next=h6_index,
            face=t2_index
        )
        
        # insert new half edges
        # self.half_edges += [h6, h7, h12, h13]
        
        # insert new triangles
        self.F = np.vstack([self.F, t2, t5])
        
        # insert new vertex
        self.V = np.vstack([self.V, v4])
        
        # save unreferenced half edges
        self.unreferenced_half_edges.extend([h0_index, h3_index])
        
        # save unreferenced triangles
        self.unreferenced_faces.extend([t0_index])  
        
        # update next on boundary
        v2_ring = self.one_ring(h11_index)
        for he_index in v2_ring:
            he = self.half_edges[he_index]
            if self.half_edges[he.twin].face==-1:
                he_twin = self.half_edges[he.twin]
                he_twin.next = h11_index

    def edge_flip(self, h0_index:int):
        he0 = self.half_edges[h0_index]
        if he0.target_bdry and not he0.source_bdry: # preference
            return self.edge_flip(h0_index=he0.twin)
        he0_twin = self.half_edges[he0.twin]
        assert (he0.face != -1 and he0_twin.face != -1), f"edge_flip({he0}) called but has a boundary"
        # get data
        
        h1_index = self.half_edges[h0_index].next
        h2_index = self.half_edges[h1_index].next
        h3_index = self.half_edges[h0_index].twin

        h4_index = self.half_edges[h3_index].next
        h5_index = self.half_edges[h4_index].next

        t0_index = self.half_edges[h0_index].face
        t1_index = self.half_edges[h3_index].face

        v2_index, v3_index = self.half_edges[h1_index].vertex_indices
        v0_index, v1_index = self.half_edges[h4_index].vertex_indices
        
        is_v1_bdry = self.half_edges[h4_index].target_bdry
        is_v3_bdry = self.half_edges[h1_index].target_bdry

        # Update Half edges
        self.update_half_edge(h0_index,next=h5_index, vertex_indices=[v3_index, v1_index], source_bdry=is_v3_bdry, target_bdry=is_v1_bdry)

        self.update_half_edge(h1_index,next=h0_index)

        self.update_half_edge(h2_index, next=h4_index, face=t1_index)

        self.update_half_edge(h3_index,next=h2_index, vertex_indices=[v1_index, v3_index], source_bdry=is_v1_bdry, target_bdry=is_v3_bdry)

        self.update_half_edge(h4_index,next=h3_index)

        self.update_half_edge(h5_index,next=h1_index, face=t0_index)

        # Update Triangles
        self.replace_triangle_by_index(t0_index, np.array([v1_index, v2_index, v3_index]))
        self.replace_triangle_by_index(t1_index, np.array([v0_index, v1_index, v3_index]))
    
    
    def edge_collapse(self, h0_index: int) -> list:    
        he0 = self.half_edges[h0_index]
        assert not (he0.source_bdry and he0.target_bdry), f"edge_collapse({he0}) called but has both source and target vertices on boundary"
        if he0.target_bdry:
            return self.edge_collapse(h0_index=he0.twin) # TODO maybe shouldn't
        # get data 
        h1_index = self.half_edges[h0_index].next
        h2_index = self.half_edges[h1_index].next
        h3_index = self.half_edges[h0_index].twin

        h4_index = self.half_edges[h3_index].next
        h5_index = self.half_edges[h4_index].next

        h6_index = self.half_edges[h2_index].twin
        h7_index = self.half_edges[h4_index].twin
        h8_index = self.half_edges[h5_index].twin
        h9_index = self.half_edges[h1_index].twin

        t0_index = self.half_edges[h0_index].face
        t1_index = self.half_edges[h3_index].face

        v0_index, v3_index = self.half_edges[h4_index].vertex_indices
        v1_index, v2_index = self.half_edges[h1_index].vertex_indices

        # updates 
        is_v0_bdry = he0.source_bdry

        v1_one_ring = self.one_ring(h3_index) # list of h_edges that have v1 as source
        p_ring = [] # all half-edges that have v1 as source, except h1 and h3

        for h_index in v1_one_ring: 
            _, v_index = self.half_edges[h_index].vertex_indices
            if v_index in (v0_index, v2_index): # if h1 or h3
                continue

            # Update triangles - in each triangle in v1_one_ring (except t0,t1) replace v1 by v0 
            self.update_triangle_by_vertex_indices(h_index, v0_index, v1_index)

            p_ring.append(h_index) 
            
            if v_index == v3_index: # skip over h8, because we don't wish to update h5, to keep it for revert
                continue

            # Update halfedges - for each he in v1_one_ring, except (h1,h3), update its and its twin vertices
            self.update_half_edge(h_index, vertex_indices=[v0_index, v_index], source_bdry=is_v0_bdry)

            th_index = self.half_edges[h_index].twin
            self.update_half_edge(th_index,vertex_indices=[v_index, v0_index], target_bdry=is_v0_bdry)

        
        # update h6.twin=h9, h9.twin=h6, h8.twin=h7, h7.twin=h8 half edge twins
        self.update_half_edge(h6_index, twin=h9_index)

        self.update_half_edge(h9_index, twin=h6_index, vertex_indices=[v2_index, v0_index], target_bdry=is_v0_bdry)

        self.update_half_edge(h7_index,twin=h8_index)

        # all updates beside twin are redundant, happen in loop, but for clarity
        self.update_half_edge(h8_index, twin=h7_index, vertex_indices=[v0_index, v3_index], source_bdry=is_v0_bdry)

        # save unreferenced half edges
        self.unreferenced_half_edges.extend([h0_index, h1_index, h2_index, h3_index, h4_index, h5_index])
        # save unreferenced triangles
        self.unreferenced_faces.extend([t0_index, t1_index])
        # save unreferenced vertex
        self.unreferenced_vertices.extend([v1_index])

        return p_ring 
    
    
    def revert_edge_collapse(self, p_ring:list):
        # p_ring contains all half edges that have v1 as source, except h1 and h3
        # get data 
        h5_index = self.unreferenced_half_edges[-1]
        h4_index = self.unreferenced_half_edges[-2]
        h2_index = self.unreferenced_half_edges[-4]
        h1_index = self.unreferenced_half_edges[-5]

        h6_index = self.half_edges[h2_index].twin
        h7_index = self.half_edges[h4_index].twin
        h8_index = self.half_edges[h5_index].twin
        h9_index = self.half_edges[h1_index].twin

        v0_index, v3_index = self.half_edges[h4_index].vertex_indices
        v1_index, v2_index = self.half_edges[h1_index].vertex_indices
        is_v1_bdry = self.half_edges[h1_index].source_bdry
        # updates 

        for h_index in p_ring:
            _, v_index = self.half_edges[h_index].vertex_indices
            # update triangle 
            self.update_triangle_by_vertex_indices(h_index, v1_index, v0_index)
            
            if v_index == v3_index: # skip over h8, because its twin is h7 currently
                continue

            # update half edge vertices
            self.update_half_edge(h_index, vertex_indices=[v1_index, v_index])

            th_index =self.half_edges[h_index].twin
            self.update_half_edge(th_index, vertex_indices=[v_index, v1_index], target_bdry=is_v1_bdry)

        # update half edge twins

        self.update_half_edge(h6_index,twin=h2_index)

        self.update_half_edge(h9_index, twin=h1_index, vertex_indices=[v2_index, v1_index], target_bdry=is_v1_bdry)

        self.update_half_edge(h7_index, twin=h4_index)

        self.update_half_edge(h8_index, twin=h5_index, vertex_indices=[v1_index, v3_index], source_bdry=is_v1_bdry)

        # remove unreferenced half edges
        self.unreferenced_half_edges = self.unreferenced_half_edges[:-6]

        # remove unreferenced triangles
        self.unreferenced_faces = self.unreferenced_faces[:-2]

        # remove unreferenced vertices
        self.unreferenced_vertices.pop()

# if __name__=="__main__":
#     from utils_load import *
#     from utils_vis import *
#     my_he_model = HalfEdgeTriMesh("hex_grid_uv_03_ccw.obj")
#     my_he_model.edge_flip(5)
#     vis_halfedge_model(my_he_model, wireframe=True, vert_labels=True, edge_labels=True, face_labels=True)
#     print('Finished')