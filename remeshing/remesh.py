

import os
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Mesh():
    def __init__(self, verts=[], faces=[]):
        self.verts: list[Vertex] = verts
        self.faces: list[Face] = faces
    
    def __init__(self, path:str):
        data = json.loads(open(path, "r").read())
        faces_index = data["faces"] # [ [f0_i0,f0_i1,f0_i2], [f1_i0,f1_i1,f1_i2], ... ,#faces]
        self.verts = [Vertex(*vert_cords) for vert_cords in data["verts"]] # [V(x0,y0,z0), V(x1,y1,z1), ... , #verts]
        self.faces = [Face([self.verts[i] for i in face]) for face in faces_index] # [F([V0,V1,V2]), F([V1,V2,V3]), ... , #faces]
        egde_counts = defaultdict(int)
        for face in self.faces:
            # iterate over edges (v1,v2) in a way that v1 will always be the smaller vertex
            for v1, v2 in zip(face.verts, face.verts[1:] + [face.verts[0]]):
                if v1.x > v2.x or (v1.x == v2.x and v1.y > v2.y) or (v1.x == v2.x and v1.y == v2.y and v1.z > v2.z):
                    v1, v2 = v2, v1
                edge = (v1, v2)
                egde_counts[edge] += 1

        for edge, count in egde_counts.items():
            if count == 1:
                for vert in edge:
                    vert.boundary = True

    def area_variance_measure(self):
        total_area = 0
        for face in self.faces:
            total_area += face.area
        average_area = total_area / len(self.faces)
        variance = 0
        for face in self.faces:
            variance += (face.area - average_area) ** 2
        return variance

    def average_edge_length(self):
        total_length = 0
        for face in self.faces:
            for v1, v2 in zip(face.verts, face.verts[1:] + [face.verts[0]]):
                total_length += np.linalg.norm(v1.np_array - v2.np_array)
        return total_length / len(self.faces) / 3
        

    def edge_length_variance_measure(self):
        average_length = self.average_edge_length()
        variance = 0
        for face in self.faces:
            for v1, v2 in zip(face.verts, face.verts[1:] + [face.verts[0]]):
                variance += (np.linalg.norm(v1.np_array - v2.np_array) - average_length) ** 2
        return variance

    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Extract vertex coordinates
        verts = [(v.x, v.y, v.z) for v in self.verts]

        # Create a list of faces with vertex coordinates
        poly3d = [[(vert.x, vert.y, vert.z) for vert in face.verts] for face in self.faces]

        # Create a Poly3DCollection object
        mesh = Poly3DCollection(poly3d, edgecolor='k', alpha=0.5)

        ax.add_collection3d(mesh)

        # Extract the limits for the plot
        x, y, z = zip(*verts)
        ax.set_xlim([min(x), max(x) + 0.1])
        ax.set_ylim([min(y), max(y) + 0.1])
        ax.set_zlim([min(z), max(z) + 0.1])
        
        # Plot all vertices, color boundary vertices in red, make them big
        for vert in self.verts:
            color = 'r' if vert.boundary else 'b'
            ax.scatter(vert.x, vert.y, vert.z, color=color, s=100)

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        plt.show()

    def remesh(self, lambda_schedule, l=None):
        if l is None:
            l = self.average_edge_length()
        l_squered = l ** 2
        print(f"{l_squered=}")
        self.visualize()
        for lambda_ in lambda_schedule:
            for face in self.faces[:]:
                if face not in self.faces:
                    continue
                for v1, v2 in zip(face.verts[:], face.verts[1:] + [face.verts[0]]):
                    length_squered = (v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2 + (v1.z - v2.z) ** 2
                    print(f"{length_squered=}")
                    if length_squered > 16 / 9 * l_squered:
                        print(f"Splitting edge {(v1, v2)}")
                        v1.split_edge(v2, self)
                    elif length_squered < 16 / 25 * l_squered:
                        print(f"Uniting edge {(v1, v2)}")
                        v1.unite(v2, self)
                    else:
                        continue
                    break
            for vert in self.verts:
                print(vert.faces)
                if len(vert.faces) <= 1:
                    continue
                for f1, f2 in zip(vert.faces[:], vert.faces[1:] + ([vert.faces[0]] if not vert.boundary else [])):
                    f1.maybe_edge_flip(f2)
            for vert in self.verts:
                if not vert.boundary:
                    vert.tangential_smoothing(lambda_)
            self.visualize()
            

class Vertex():
    def __init__(self, x:float, y:float, z:float = 0):
        self.x = x
        self.y = y
        self.z = z
        self.faces: list[Face] = []
        self.boundary = False
        
    def __add__(self, other):
        return Vertex(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __itruediv__(self, other:float):
        self.x /= other
        self.y /= other
        self.z /= other
        return self
        
    def __repr__(self):
        return f"V({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
    
    @property
    def np_array(self) -> np.array:
        return np.array([self.x, self.y, self.z])

    @property
    def area(self):
        area = 0
        for face in self.faces:
            area += face.area
        return area / 3

    @property
    def normal(self):
        normal = np.array([0, 0, 0], dtype=float)
        for face in self.faces:
            normal += face.weighted_normal
        return normal / np.linalg.norm(normal)
    
    @property
    def factored_valance(self):
        return len(self.faces) + (3 if self.boundary else 0)

    def split_edge(self, other:'Vertex', mesh:Mesh):
        mid_point = self + other
        mid_point /= 2.0
        for face in self.faces[:]:
            if face in other.faces:
                # Move the face to the mid point
                face_copy = Face.Copy(face)
                face.switch(self, mid_point)
                face_copy.switch(other, mid_point)
                mesh.faces.append(face_copy)
        mesh.verts.append(mid_point)

    def unite(self, other:'Vertex', mesh:Mesh):
        self.x = (self.x + other.x) / 2
        self.y = (self.y + other.y) / 2
        self.z = (self.z + other.z) / 2
        for face in other.faces[:]:
            if self in face.verts:
                face.delete()
                mesh.faces.remove(face)
                continue
            face.verts[face.verts.index(other)] = self
            self.faces.append(face)
        other.faces = []
        mesh.verts.remove(other)
    
    def tangential_smoothing(self, lambda_:float):
        gravity = 0
        centroid = np.array([0, 0, 0], dtype=float)
        for face in self.faces:
            for vert in face.verts:
                a = vert.area
                centroid += a * vert.np_array
                gravity += a
        centroid /= gravity
        update = lambda_ * (centroid - self.np_array)
        normal = self.normal
        update = update - np.dot(update, normal) * normal
        self.x, self.y, self.z = self.x + update[0], self.y + update[1], self.z + update[2]

class Face():
    def __init__(self, verts):
        self.verts: list[Vertex] = verts
        for vert in verts:
            vert.faces.append(self)
    
    def delete(self):
        for vert in self.verts:
            vert.faces.remove(self)
    
    def __repr__(self):
        return f"F({self.verts})"

    @staticmethod
    def Copy(other:'Face'):
        return Face(other.verts[:])

    @property
    def weighted_normal(self):
        v1, v2, v3 = self.verts

        # Compute the vectors representing two sides of the triangle
        v4 = np.array([v1.x - v3.x, v1.y - v3.y, v1.z - v3.z])
        v5 = np.array([v2.x - v3.x, v2.y - v3.y, v2.z - v3.z])
        
        # Compute the cross product of these vectors
        return np.cross(v4, v5)

    @property
    def area(self):
        return 0.5 * np.linalg.norm(self.weighted_normal)

    def switch(self, old_vertex, new_vertex):
        index = self.verts.index(old_vertex)
        self.verts[index].faces.remove(self)
        self.verts[index] = new_vertex
        self.verts[index].faces.append(self)

    def maybe_edge_flip(self, other):
        other_vert = [vert for vert in other.verts if vert not in self.verts][0]
        my_vert = [vert for vert in self.verts if vert not in other.verts][0]
        if len([vert for vert in self.verts if vert in other.verts]) < 2:
            return
        first_shared_vert, second_shared_vert = [vert for vert in self.verts if vert in other.verts]
        if first_shared_vert.factored_valance + second_shared_vert.factored_valance > 1 + my_vert.factored_valance + other_vert.factored_valance:
            print(f"Flipping edge {first_shared_vert} - {second_shared_vert}")
            self.switch(first_shared_vert, other_vert)
            other.switch(second_shared_vert, my_vert)


if __name__ == "__main__":             
    import time
    from contextlib import contextmanager
    @contextmanager
    def timeit():
        start = time.time()
        yield
        print(f"Time: {time.time() - start:.2f}")

    import sys
    mesh = Mesh(os.path.join(".", "samples", sys.argv[1] + ".json"))
    with timeit():
        print(mesh.area_variance_measure())
        print(mesh.edge_length_variance_measure())
        if "remesh" in sys.argv:
            mesh.remesh([0.0])
            print(mesh.area_variance_measure())
            print(mesh.edge_length_variance_measure())
    mesh.visualize()
