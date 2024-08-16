import numpy as np
import json
import os 

SAMPLES_PATH = "/home/ehud/technion/surf/remeshing/samples/"

def get_np_V_F(filename):
    # V (n,3) float64,
    # F (m,3) int32
    if filename.endswith('.json'):
        return get_np_V_F_from_json(filename)
    elif filename.endswith('.obj'):
        return get_np_V_F_from_obj(filename)

def get_np_V_F_from_json(json_filename):
    json_path = os.path.join(SAMPLES_PATH, json_filename)
    with open(json_path, 'r') as file:
        data = json.load(file)
    vertices = np.array(data['verts'], dtype=np.double) # shape (#vertices, 3)
    triangles = np.array(data['faces'], dtype=np.int32) # shape (#triangles, 3)
    
    return vertices, triangles

def get_np_V_F_from_obj(obj_filename):
    obj_path = os.path.join(SAMPLES_PATH, obj_filename)
    vertices, faces = [], []
    with open(obj_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
            elif line.startswith('f '):
                parts = line.split()
                face = [int(parts[1].split('/')[0]) - 1, 
                        int(parts[2].split('/')[0]) - 1, 
                        int(parts[3].split('/')[0]) - 1]
                faces.append(face)
    vertices_np = np.array(vertices, dtype=np.float64) #(v,3)
    faces_np = np.array(faces, dtype=np.int32) #(f,3)
    return vertices_np, faces_np