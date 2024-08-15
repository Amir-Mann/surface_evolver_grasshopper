from typing import List, Union
import numpy as np
import numpy.typing as npt
import open3d as o3d
from open3d.geometry import TriangleMesh # type: ignore
from vector_tools import BiPoint

class MyHalfEdge:
    def __init__(self, 
                 idx: int  =-1,
                 next: int =-1,
                 twin: int =-1,
                 face: int =-1,
                 vertex_indices: List[int] = [-1,-1]):
        self.idx = idx
        self.twin = twin
        self.next = next
        self.face = face
        # if vertex_indices are tuple, turn to list
        if isinstance(vertex_indices, tuple):
            vertex_indices = list(vertex_indices)
        self.vertex_indices = vertex_indices
    def __repr__(self) -> str:
        return f"HE{self.idx}({self.vertex_indices}),twin={self.twin},next={self.next},face={self.face}"
        
class HalfEdgeTriMesh:
    def __init__(self,
                 V : npt.NDArray[np.float64], # shape (#vertices,3)
                 F : npt.NDArray[np.int32]): # shape (#faces,3)
        half_edges_d = dict()
        self.half_edges = []
        self.V = V # np (#verts,3) float64
        self.F = F # np (#faces,3) int32
        self.unreferenced_vertices = []
        self.unreferenced_triangles = []
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
            he_ij = MyHalfEdge(idx=ij_idx, next=jk_idx, face=face_idx, vertex_indices=eij)
            he_jk = MyHalfEdge(idx=jk_idx, next=ki_idx, face=face_idx, vertex_indices=ejk)
            he_ki = MyHalfEdge(idx=ki_idx, next=ij_idx, face=face_idx, vertex_indices=eki)
            self.half_edges.extend([he_ij, he_jk, he_ki])
            half_edges_d[eij] = he_ij
            half_edges_d[ejk] = he_jk
            half_edges_d[eki] = he_ki
        
        # populate the twins
        to_be_filled = []
        original_items = list(half_edges_d.items())
        for edge, he in original_items:
            twin_edge = (edge[1], edge[0])
            twin_he = half_edges_d.get(twin_edge, None)
            if twin_he:
                he.twin = twin_he.idx
            else: # create boundary half-edge
                twin_idx = len(self.half_edges)
                he.twin = twin_idx
                he_next = self.half_edges[he.next]
                first_vertex = he.vertex_indices[0]
                third_vertex = he_next.vertex_indices[1]
                twin_next_edge = (first_vertex, third_vertex)
                twin_next = half_edges_d.get(twin_next_edge, None)
                twin_next_idx = -1
                if twin_next:    
                    twin_next_idx = twin_next.idx
                else: # new half-edge.next will be filled later
                    to_be_filled.append((twin_idx, twin_next_edge))
                twin_he = MyHalfEdge(idx=twin_idx, twin=he.idx, next=twin_next_idx, face=-1, vertex_indices=twin_edge)
                half_edges_d[twin_edge] = twin_he
                self.half_edges.append(twin_he)
        for twin_idx, twin_next_edge in to_be_filled:
            twin_next = half_edges_d[twin_next_edge]
            self.half_edges[twin_idx].next = twin_next.idx
            
    def create_half_edge(self, next:int=-1, face:int=-1, twin:int=-1, vertex_indices:list=[-1, -1]):
        he_idx = len(self.half_edges)
        h = MyHalfEdge(idx=he_idx, next=next, face=face, twin=twin, vertex_indices=vertex_indices)
        self.half_edges.append(h)
        return h
    
    def update_half_edge(self, h_index:int, next = None, face = None, twin = None, vertex_indices = None):
        if next: self.half_edges[h_index].next = next
        if face: self.half_edges[h_index].face = face
        if twin: self.half_edges[h_index].twin = twin
        if vertex_indices: self.half_edges[h_index].vertex_indices = vertex_indices
        
    
    def get_vertices_by_indices(self, indices):
        # return (n,3) float64 for n=len(indices)
        return self.V[indices]
    
    def get_triangles_by_indices(self, indices: Union[int, List[int]]):
        return self.F[indices] # .copy()
    
    def get_face_midpoint(self, t_idx):
        indices = self.F[t_idx]
        vertices_t = self.get_vertices_by_indices(indices)
        return np.mean(vertices_t, axis=0)
    
    def get_clean_trimesh(self):
        vertices = o3d.utility.Vector3dVector(self.V)
        faces = o3d.utility.Vector3iVector(self.F)
        triangle_mesh = TriangleMesh(vertices, faces)
        triangle_mesh.remove_triangles_by_index(self.unreferenced_triangles)
        triangle_mesh.remove_vertices_by_index(self.unreferenced_vertices)
        return triangle_mesh
    
    def get_vertex_indices(self, h_index:int):
        return self.half_edges[h_index].vertex_indices
    
    def get_vertices_by_edge(self, h_index):
        indices = self.get_vertex_indices(h_index)
        return self.get_vertices_by_indices(indices)
    
    def get_next_index(self, h_index):
        return self.half_edges[h_index].next
    
    def get_twin_index(self, h_index):
        return self.half_edges[h_index].twin
    
    def get_face(self, h_index):
        return self.half_edges[h_index].face
    
    def num_half_edges(self, clean:bool = False):
        return len(self.half_edges) - len(self.unreferenced_half_edges) if clean else len(self.half_edges)
    
    def num_vertices(self, clean:bool = False):
        return len(self.V) - len(self.unreferenced_vertices) if clean else len(self.V)
    
    def num_faces(self, clean:bool = False):
        return len(self.F) - len(self.unreferenced_triangles) if clean else len(self.F)
    
   
    def get_edge_midpoint(self, h_index):
        vertices = self.get_vertices_by_edge(h_index)
        return np.mean(vertices, axis=0)
    
    def get_edges_midpoints(self):
        E = self.num_half_edges()
        seen_edges = set()
        midpoints_d = dict()
        for h_index in range(E):
            if h_index in self.unreferenced_half_edges or h_index in seen_edges:
                continue
            
            midpoints_d[h_index] = self.get_edge_midpoint(h_index)
            seen_edges.add(h_index)
            seen_edges.add(self.get_twin_index(h_index))
        return midpoints_d
    
    def get_triangles_midpoints(self):
        F = self.num_faces()
        midpoints_d = dict()
        for t_idx in range(F):
            if t_idx in self.unreferenced_triangles:
                continue
            midpoints_d[t_idx] = self.get_face_midpoint(t_idx)
        return midpoints_d
    
    
    # set vertices and triangles methods
    def replace_triangle_by_index(self, index: int, new_triangle: np.ndarray):
        self.F[index] = new_triangle
    
    def update_triangle_by_vertex_indices(self, h_index:int, v_out:int, v_target:int):
        t_index = self.half_edges[h_index].face
        triangle = self.get_triangles_by_indices(t_index) #(3,)
        triangle[triangle==v_target] = v_out
        self.replace_triangle_by_index(t_index, triangle)
    
    # ring operations
    def get_next_index_on_ring(self, h_index:int, clockwise:bool):
        if clockwise:
            th_index = self.get_twin_index(h_index)
            return self.get_next_index(th_index)
        else:
            nh_index = self.get_next_index(h_index)
            nnh_index = self.get_next_index(nh_index)
            return self.get_twin_index(nnh_index)
    
    def one_ring(self, h_index:int, clockwise:bool=True):
        # return half_edges whose starting vertices is like h_index's. returns in clockwise manner
        last = h_index
        while True:
            nh_index = self.get_next_index_on_ring(h_index, clockwise)
            yield nh_index
            if nh_index == last:
                break
            h_index = nh_index
    
    # half-edge modifying operations
    
    def edge_split(self, h0_index:int):
        h0_twin_index = self.get_twin_index(h0_index)
        assert self.half_edges[h0_index].face != -1 and self.half_edges[h0_twin_index].face != -1, f"edge_split({h0_index}) called but is boundary"
            

        E = self.num_half_edges()
        T = self.num_faces()
        V = self.num_vertices()

        # get data

        h1_index = self.get_next_index(h0_index)
        h2_index = self.get_next_index(h1_index)
        h3_index = self.get_twin_index(h0_index)

        h4_index = self.get_next_index(h3_index)
        h5_index = self.get_next_index(h4_index)

        t0_index = self.get_face(h0_index)
        t1_index = self.get_face(h3_index)

        v0_index, v1_index = self.get_vertex_indices(h4_index)
        v2_index, v3_index = self.get_vertex_indices(h1_index)

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

        # create new vertex index

        v4_index = V+0

        # create new half edges

        h6 = self.create_half_edge(
            next=h7_index,
            face=t2_index,
            twin=h8_index,
            vertex_indices=[v0_index, v4_index]
        )

        h7 = self.create_half_edge(
            next=h2_index,
            face=t2_index,
            twin=h12_index,
            vertex_indices=[v4_index, v3_index]
        )

        h8 = self.create_half_edge(
            next=h4_index,
            face=t3_index,
            twin=h6_index,
            vertex_indices=[v4_index, v0_index]
        )

        h9 = self.create_half_edge(
            next=h8_index,
            face=t3_index,
            twin=h10_index,
            vertex_indices=[v1_index, v4_index]
        )

        h10 = self.create_half_edge(
            next=h5_index,
            face=t4_index,
            twin=h9_index,
            vertex_indices=[v4_index, v1_index]
        )

        h11 = self.create_half_edge(
            next=h10_index,
            face=t4_index,
            twin=h13_index,
            vertex_indices=[v2_index, v4_index]
        )

        h12 = self.create_half_edge(
            next=h13_index,
            face=t5_index,
            twin=h7_index,
            vertex_indices=[v3_index, v4_index]
        )

        h13 = self.create_half_edge(
            next=h1_index,
            face=t5_index,
            twin=h11_index,
            vertex_indices=[v4_index, v2_index]
        )


        # create new triangles
        t2 = np.array([v4_index, v3_index, v0_index])
        t3 = np.array([v4_index, v0_index, v1_index])
        t4 = np.array([v4_index, v1_index, v2_index])
        t5 = np.array([v4_index, v2_index, v3_index])

        # create new vertex
        vertices = self.get_vertices_by_indices([v0_index, v2_index])
        v4 = BiPoint.midpoint(vertices)

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
        self.unreferenced_triangles.extend([t0_index, t1_index])
      
    def edge_split_boundary(self, h0_index:int):
        h0 = self.half_edges[h0_index]
        h3 = self.half_edges[h0.twin]
        assert h0.face==-1 or h3.face==-1, f"edge_split_boundary({h0_index}) called but isn't boundary"
        if h0.face==-1: # prefer to run over inside half-edge than boundary
            h0_index = h0.twin 
            h0, h3 = h3, h0
        E = self.num_half_edges()
        T = self.num_faces()
        V = self.num_vertices()
        
        # get data
        h1_index = self.get_next_index(h0_index)
        h2_index = self.get_next_index(h1_index)
        h3_index = h3.idx
        
        t0_index = h0.face
        
        v0_index, v2_index = self.get_vertex_indices(h0_index)
        _, v3_index = self.get_vertex_indices(h1_index)
        
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
            vertex_indices=[v0_index,v4_index]
        )
        h7 = self.create_half_edge(
            next=h2_index,
            face=t2_index,
            twin=h12_index,
            vertex_indices=[v4_index,v3_index]
        )
        h8 = self.create_half_edge(
            next=self.get_twin_index(h2_index),
            face=-1,
            twin=h6_index,
            vertex_indices=[v4_index,v0_index]
        )
        h11 = self.create_half_edge(
            next=h8_index,
            face=-1,
            twin=h13_index,
            vertex_indices=[v2_index,v4_index]
        )
        h12 = self.create_half_edge(
            next=h13_index,
            face=t5_index,
            twin=h7_index,
            vertex_indices=[v3_index,v4_index]
        )
        h13 = self.create_half_edge(
            next=h1_index,
            face=t5_index,
            twin=h11_index,
            vertex_indices=[v4_index,v2_index]
        )
        
        # create new triangles
        t2 = np.array([v4_index, v3_index, v0_index])
        t5 = np.array([v4_index, v2_index, v3_index])
        
        # create new vertex
        vertices = self.get_vertices_by_indices([v0_index, v2_index])
        v4 = BiPoint.midpoint(vertices)
        
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
        self.unreferenced_triangles.extend([t0_index])  

    def edge_collapse(self, h0_index: int) -> list:    
        # get data 
        h1_index = self.get_next_index(h0_index)
        h2_index = self.get_next_index(h1_index)
        h3_index = self.get_twin_index(h0_index)

        h4_index = self.get_next_index(h3_index)
        h5_index = self.get_next_index(h4_index)

        h6_index = self.get_twin_index(h2_index)
        h7_index = self.get_twin_index(h4_index)
        h8_index = self.get_twin_index(h5_index)
        h9_index = self.get_twin_index(h1_index)

        t0_index = self.half_edges[h0_index].face
        t1_index = self.half_edges[h3_index].face

        v0_index, v3_index = self.get_vertex_indices(h4_index)
        v1_index, v2_index = self.get_vertex_indices(h1_index)

        # updates 

        v1_ring = self.one_ring(h3_index)
        p_ring = []

        for h_index in v1_ring: #

            _, v_index = self.get_vertex_indices(h_index)

            if v_index in (v0_index, v2_index):
                continue

            # update triangle 

            self.update_triangle_by_vertex_indices(h_index, v0_index, v1_index)

            p_ring.append(h_index) 

            if v_index == v3_index:
                continue

            # update half edge vertices

            self.update_half_edge(
                h_index,
                vertex_indices=[v0_index, v_index]
            )

            th_index = self.get_twin_index(h_index)
            self.update_half_edge(
                th_index,
                vertex_indices=[v_index, v0_index]
            )

        # update half edge twins

        self.update_half_edge(
            h6_index,
            twin=h9_index
        )

        self.update_half_edge(
            h9_index,
            twin=h6_index,
            vertex_indices=[v2_index, v0_index]
        )

        self.update_half_edge(
            h7_index,
            twin=h8_index
        )

        self.update_half_edge(
            h8_index,
            twin=h7_index,
            vertex_indices=[v0_index, v3_index]
        )

        # save unreferenced half edges
        self.unreferenced_half_edges.extend([h0_index, h1_index, h2_index, h3_index, h4_index, h5_index])

        # save unreferenced triangles
        self.unreferenced_triangles.extend([t0_index, t1_index])

        # save unreferenced vertex
        self.unreferenced_vertices.extend([v1_index])

        return p_ring 

if __name__=="__main__":
    from utils_load import *
    from utils_vis import *
    vertices_np, triangles_np = load_np("hex_grid_uv_03_ccw.obj")
    my_he_model = HalfEdgeTriMesh(vertices_np, triangles_np)
    my_he_model.edge_collapse(29)
    vis_halfedge_model(my_he_model, wireframe=True, vert_labels=True, edge_labels=True, face_labels=True)
    print('Finished')