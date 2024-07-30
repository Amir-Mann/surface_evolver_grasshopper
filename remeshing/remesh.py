

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
        ax.set_xlim([min(x), max(x)])
        ax.set_ylim([min(y), max(y)])
        ax.set_zlim([min(z), max(z)])
        
        # Plot all vertices, color boundary vertices in red, make them big
        for vert in self.verts:
            color = 'r' if vert.boundary else 'b'
            ax.scatter(vert.x, vert.y, vert.z, color=color, s=100)

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        plt.show()



class Vertex():
    def __init__(self, x:float, y:float, z:float = 0):
        self.x = x
        self.y = y
        self.z = z
        self.faces: list[Face] = []
        self.boundary = False
    
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
        
    
    def __add__(self, other):
        return Vertex(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __idiv__(self, other:float):
        self.x /= other
        self.y /= other
        self.z /= other

    def split_edge(self, other:'Vertex', mesh:Mesh):
        mid_point = self + other
        mid_point /= 2
        for face in self.faces:
            if face in other.faces:
                # Move the face to the mid point
                face_copy = Face.Copy(face)
                face.switch(self, mid_point)
                face_copy.switch(other, mid_point)
                mesh.faces.append(face_copy)
        mesh.verts.append(mid_point)
        



    def __repr__(self):
        return f"V({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

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

    def switch(self, old_vertex, new_vertex):
        index = self.verts.index(old_vertex)
        self.verts[index].faces.remove(self)
        self.verts[index] = new_vertex
        self.verts[index].faces.append(self)


trig_mesh = Mesh("./samples/trig.json")
two_trig_mesh = Mesh("./samples/two_trig.json")
hex_grid_mesh = Mesh("./samples/hex_grid.json")
hex_grid_mesh.visualize()
# sample1_path = "./samples/sample1.json"
# mesh_sample1 = Mesh(sample1_path)
# data = json.loads(open("./samples/sample1.json", "r").read())
# faces_index = data["faces"]
# verts = [Vertex(*vert_cords) for vert_cords in data["verts"]]
# faces = [Face([verts[i] for i in face]) for face in faces_index]
# egde_counts = {}
# for face in faces:
#     for v1, v2 in zip(face.verts, face.verts[1:] + [face.verts[0]]):
#         edge = (v1, v2)
#         if edge in egde_counts:
#             egde_counts[edge] += 1
#         else:
#             egde_counts[edge] = 1

# for edge, count in egde_counts.items():
#     if count == 1:
#         for vert in edge:
#             vert.boundary = True

# vert1 = faces[0].verts[0]
# for i, face in enumerate(faces):
#     if vert1 in face.verts:
#         print(i, face)

# vert1.x = 0.5
# vert1.y = 0.5
# vert1.z = 0.5 

# for i, face in enumerate(faces):
#     if vert1 in face.verts:
#         print(i, face)
