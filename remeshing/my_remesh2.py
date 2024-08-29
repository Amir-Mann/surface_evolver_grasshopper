import os
import numpy as np 
from HalfEdge.half_edge import HalfEdgeTriMesh
from HalfEdge.utils_math import angle, is_collinear
import HalfEdge.utils_he as utils_he
import threading

class IsotropicRemesher:
    def __init__(self, model: HalfEdgeTriMesh):
        self.model = model
        
    @property
    def half_edges(self):
        return self.model.half_edges

    @property
    def V(self):
        return self.model.V
    
    @property
    def F(self):
        return self.model.F
    
    def visualize(self, *args, **kwargs):
        self.model.visualize(*args, **kwargs)
    
    def has_foldover_triangles(self, normals_pre, normals_pos, threshold=np.pi/6):
        for normals in zip(normals_pre, normals_pos):
            if angle(normals) > threshold:
                return True
        return False

    def has_collinear_edges(self, h0_index: int):
        _, h2_index, _, h5_index = self.model.adjacent_half_edges(h0_index) # h1, h2, h4, h5

        v2_index, v0_index = self.half_edges[h2_index].vertex_indices
        v3_index, v1_index = self.half_edges[h5_index].vertex_indices

        vertices = self.V[[v1_index, v2_index, v3_index]]
        if is_collinear(vertices): 
            return True
        he0 = self.half_edges[h0_index] # todo debug
        vertices2 = self.V[[v0_index, v2_index, v3_index]]
        if is_collinear(vertices2): 
            return True

        return False
    
    def has_small_edges(self, h_index:int, L_high:float):
        # check that new edges are smaller than high.
        # if this is true than we don't collapse edge
        v0_index = self.half_edges[h_index].vertex_indices[0]
        th_index = self.half_edges[h_index].twin
        v1_index_ring = self.model.vertex_index_ring(th_index)
        for vi in v1_index_ring:
            # go over each of the edges that will be created if the collapse happens
            # and check if the length is smaller than L_high
            vs = self.V[[vi, v0_index]] # (2,3) float64
            edge_length = np.linalg.norm(vs[0] - vs[1])
            if edge_length > L_high:
                return False
        return True 

    def has_collapse_connectivity(self, h_index:int):
        v0_ring = set(self.model.vertex_index_ring(h_index))
        th_index = self.half_edges[h_index].twin
        v1_ring = set(self.model.vertex_index_ring(th_index))
        return len(v0_ring.intersection(v1_ring)) == 2

    def has_flip_connectivity(self, h0_index: int):
        h0_valence = self.model.valence(h0_index)
        if  h0_valence == 3:
            return False
        
        h3_index = self.half_edges[h0_index].twin
        h3_valence = self.model.valence(h3_index)

        if h3_valence == 3:
            return False

        return True

    def split_long_edges(self, threshold: float):
        
        n = 0
        E = len(self.model.half_edges)

        for h0_index in range(E):

            if h0_index in self.model.unreferenced_half_edges:
                continue

            h0_edge_len = self.model.edge_len(h0_index)
            if  h0_edge_len <= threshold:
                continue
            he0_index = self.model.half_edges[h0_index]
            self.model.edge_split(h0_index)

            n += 1

        return n 

    def collapse_short_edges(self, L_low, L_high, foldover:float=0):
        n = 0
        E = len(self.model.half_edges)
        skip = []

        for h0_index in range(E):

            # skip if twin already tested

            h3_index = self.half_edges[h0_index].twin

            if h3_index in skip:
                continue
            
            skip.append(h3_index)

            if h0_index in self.model.unreferenced_half_edges:
                continue
    

            h0_edge_len = self.model.edge_len(h0_index) # TODO debug
            if self.model.edge_len(h0_index) >= L_low:
                continue

            if not self.has_small_edges(h0_index, L_high):
                continue

            # https://stackoverflow.com/questions/27049163/mesh-simplification-edge-collapse-conditions
            if not self.has_collapse_connectivity(h0_index):
                continue

            # Compute normals before collapse  # TODO
            if foldover > 0: normals_pre = list(self.model.normal_ring(h3_index))[1:-1] # exclude f0 and f1
            # TODO
            he0_index = self.model.half_edges[h0_index]
            if he0_index.source_bdry and he0_index.target_bdry:
                continue
            p_ring = self.model.edge_collapse(h0_index) 

            # Compute normals after collapse 
            if foldover > 0:
                 
                normals_pos = map(self.model.normal, p_ring)
                if self.has_foldover_triangles(normals_pre, normals_pos, threshold=foldover):
                    self.model.revert_edge_collapse(p_ring)
                    continue

            n += 1

        return n

    def equalize_valences(self, sliver:bool=False, foldover:float=0):

        n = 0
        E = len(self.half_edges)
        skip = []

        for h0_index in range(E):
            # skip if twin already tested
            h3_index = self.half_edges[h0_index].twin

            if h3_index in skip:
                continue
            
            skip.append(h3_index)
                        
            if h0_index in self.model.unreferenced_half_edges:
                continue
            
            he0 = self.model.half_edges[h0_index]
            # if boundary edge
            if he0.face==-1 or self.model.half_edges[h3_index].face==-1:
                continue
            if not self.has_flip_connectivity(h0_index):
                continue 

            if self.has_collinear_edges(h0_index):
                continue 

            # Compute normals before flip
            if foldover > 0: normals_pre = ( self.model.normal(h0_index), self.model.normal(h3_index) )

            if sliver: deviation_compactness_pre = self.compactness_deviation(h0_index)

            deviation_valence_pre = self.valence_deviation(h0_index)
            
            # todo 

            is_he0_bdry = he0.face == -1 or self.model.half_edges[h3_index].face == -1
            if is_he0_bdry:
                continue
            self.model.edge_flip(h0_index)
            
            if foldover > 0:
                normals_pos = ( self.model.normal(h0_index), self.model.normal(h3_index) )

                if self.has_foldover_triangles(normals_pre, normals_pos, threshold=foldover):
                    self.model.edge_flip(h0_index)
                    continue

            deviation_valence_pos = self.valence_deviation(h0_index)

            if deviation_valence_pre < deviation_valence_pos:
                self.model.edge_flip(h0_index)
                continue
            # if (deviation_valence_pos < deviation_valence_pre):
                # print(f"flip of edge {h0_index} was successful")
            
            if sliver:
                deviation_compactness_pos = self.compactness_deviation(h0_index)

                if deviation_compactness_pre < deviation_compactness_pos:
                    self.model.edge_flip(h0_index)
                    continue

            n += 1

        return n

    def vertex_relocation(self, iter: int, num_iters: int):
        E = len(self.model.half_edges)
        skip = []
        # lambda_ = 1.0 if iter < num_iters/2 else 0.75
        lambda_ = 0.9
        if iter > num_iters/2:
            lambda_ /= 2
        # if iter > 2*num_iters/3:
        #     lambda_ /= 2
        
        for h_index in range(E):

            if h_index in self.model.unreferenced_half_edges:
                continue
            
            v_index = self.half_edges[h_index].vertex_indices[0]

            if v_index in skip:
                continue 

            if v_index in self.model.unreferenced_vertices:
                continue

            
            self.tangential_smoothing(h_index, lambda_)
            
            skip.append(v_index)

        # return new_vertices

    def isotropic_remeshing(self, L:float, num_iters:int=20, foldover:float=0, sliver:bool=False):
        L_low, L_high = 4/5.*L, 4/3.*L
        for iter in range(num_iters): 

            print(10*'=' + f' ITER {iter+1}/{num_iters} ' + 10*'=')

            s = self.split_long_edges(L_high)
            print(f'split long edges ({s})')

            c = self.collapse_short_edges(L_low, L_high, foldover)
            print(f'collapse short edges ({c})')
            
            f = self.equalize_valences(sliver, foldover)
            print(f'flip edges ({f})')

            self.vertex_relocation(iter=iter, num_iters=num_iters)
            print(f'tangential smoothing')
            
            # std_edge_len = utils_he.std_deviation_edge_len(he_trimesh)
            # std_area, relative_mean_area_error = utils_he.face_area_stats(he_trimesh)
            # print(f"{iter+1}/{num_iters}, std. deviation edge len: {std_edge_len:.2f}, std. deviation face area: {std_area:.2f}, relative mean area error: {relative_mean_area_error:.2f}")

    def tangential_smoothing(self, h_index, lambda_:float):
        he0 = self.model.half_edges[h_index]
        if he0.source_bdry:
            return
        v0_idx = self.half_edges[h_index].vertex_indices[0]
        v0_pos = self.model.get_start_vertex_by_edge(h_index)
        vertex_ring, weighted_normal_ring = self.model.gravity_ring(h_index) # (n,3) float64, (n,3) float64
        area_ring = np.array([np.linalg.norm(v) for v in weighted_normal_ring]) # (n,) float64
        gravity = np.sum(area_ring)
        centroid = np.sum([v * a for v, a in zip(vertex_ring, area_ring)], axis=0) # (3,) float64
        centroid /= gravity
        #  update
        update = lambda_ * (centroid - v0_pos)
        # normal as average of face normals
        normal = np.mean(weighted_normal_ring, axis=0)
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            return
        normal /= normal_norm
        update = update - np.dot(update, normal) * normal

        new_v0_pos = v0_pos + update
        self.model.V[v0_idx] = new_v0_pos
        pass
        

    def compactness_deviation(self, h_index:int):
        return sum([1.-np.mean(list(self.model.compactness_ring(i))) for i in self.model.adjacent_half_edges(h_index)])

    def valence_deviation(self, h_index:int):
        h1_idx, h2_idx, h4_idx, h5_idx = self.model.adjacent_half_edges(h_index)
        h1_valence = self.model.valence(h1_idx)
        h2_valence = self.model.valence(h2_idx)
        h4_valence = self.model.valence(h4_idx)
        h5_valence = self.model.valence(h5_idx)
        # return sum([abs(self.model.valence(i)-6) for i in self.model.adjacent_half_edges(h_index)])
        return sum([abs(h1_valence-6), abs(h2_valence-6), abs(h4_valence-6), abs(h5_valence-6)])


if __name__=="__main__":
    # model_name = "hex_grid_uv_03_ccw.obj"
    # model_name = "sample2.json"
    # model_name = "iphi_bad10k.off"
    model_name = "wolf_head.obj"
    he_trimesh = HalfEdgeTriMesh.from_model_path(model_name)
    # he_trimesh.visualize(v_labels=False, e_labels=False, f_labels=False)
    # L = he_trimesh.get_percentile_edge_length(0.1)
    L = 0.9 *he_trimesh.get_average_edge_length()
    L_low, L_high = 4/5.*L, 4/3.*L
    
    sliver = True
    foldover=np.pi/9
    num_iters=20
    run_stats= f'L low = {L_low:.2f} L target = {L:.2f} L high = {L_high:.2f}, foldover = {foldover:.2f}, sliver = {sliver}'
    
    utils_he.save_stats(he_trimesh, prefix="before", rewrite=True, extra=run_stats)
    remesher = IsotropicRemesher(he_trimesh)
    remesher.isotropic_remeshing(L=L, num_iters=num_iters, foldover=foldover, sliver=sliver)
    utils_he.save_stats(he_trimesh, prefix=f"after {num_iters} iters", rewrite=False)
    
    # he_trimesh.visualize(v_labels=False, e_labels=False, f_labels=False)
    # remesher.visualize()
    remesher.model.save_to_obj(open_in_meshlab=True)
    # os.system('play -nq -t alsa synth 0.7 sine 440')
    