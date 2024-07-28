

import json

class Vertex():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.faces = []
        self.boundary = False
    
    def unite(self, other):
        self.x = (self.x + other.x) / 2
        self.y = (self.y + other.y) / 2
        self.z = (self.z + other.z) / 2
        for face in other.faces[:]:
            if self in face.verts:
                face.delete()
                continue
            face.verts[face.verts.index(other)] = self
            self.faces.append(face)
        other.faces = []
    
    def __repr__(self):
        return f"V({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

class Face():
    def __init__(self, verts):
        self.verts = verts
        for vert in verts:
            vert.faces.append(self)
    
    def delete(self):
        for vert in self.verts:
            vert.faces.remove(self)
    
    def __repr__(self):
        return f"F({self.verts})"


data = json.loads(open("./samples/sample1.json", "r").read())
faces_index = data["faces"]
verts = [Vertex(*vert_cords) for vert_cords in data["verts"]]
faces = [Face([verts[i] for i in face]) for face in faces_index]
egde_counts = {}
for face in faces:
    for v1, v2 in zip(face.verts, face.verts[1:] + [face.verts[0]]):
        edge = (v1, v2)
        if edge in egde_counts:
            egde_counts[edge] += 1
        else:
            egde_counts[edge] = 1

for edge, count in egde_counts.items():
    if count == 1:
        for vert in edge:
            vert.boundary = True

vert1 = faces[0].verts[0]
for i, face in enumerate(faces):
    if vert1 in face.verts:
        print(i, face)

vert1.x = 0.5
vert1.y = 0.5
vert1.z = 0.5 

for i, face in enumerate(faces):
    if vert1 in face.verts:
        print(i, face)
