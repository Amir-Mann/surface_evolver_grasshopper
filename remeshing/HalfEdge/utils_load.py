import numpy as np
import json
import os 

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(base_dir, 'samples')
OUT_DIR = os.path.join(base_dir, 'out')

def get_np_V_F(filename):
    # V (n,3) float64,
    # F (m,3) int32
    model_name = filename.split('.')[0]
    if filename.endswith('.json'):
        v,f = get_np_V_F_from_json(filename)
    elif filename.endswith('.obj'):
        v, f = get_np_V_F_from_obj(filename)
    elif filename.endswith('.off'):
        v, f = get_np_V_F_from_off(filename)
    return v, f, model_name

def get_np_V_F_from_json(json_filename):
    json_path = os.path.join(SAMPLES_DIR, json_filename)
    with open(json_path, 'r') as file:
        data = json.load(file)
    vertices = np.array(data['verts'], dtype=np.double) # shape (#vertices, 3)
    triangles = np.array(data['faces'], dtype=np.int32) # shape (#triangles, 3)
    
    return vertices, triangles

def get_np_V_F_from_obj(obj_filename):
    obj_path = os.path.join(SAMPLES_DIR, obj_filename)
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

def get_np_V_F_from_off(off_filename):
    def strip_comments(line):
        # Remove any inline comments starting with # or //
        line = line.split('#')[0]
        line = line.split('//')[0]
        return line.strip()
    off_path = os.path.join(SAMPLES_DIR, off_filename)
    with open(off_path, 'r') as file:
        # Read the first line (OFF header)
        header = file.readline().strip()
        if header != "OFF":
            raise ValueError("The file is not in OFF format.")

        # Skip comments and empty lines until the data line
        while True:
            line = strip_comments(file.readline())
            if not line:
                continue
            else:
                # Read the number of vertices, faces, and (optionally) edges
                n_verts, n_faces, _ = map(int, line.split())
                break

        # Read the vertices
        vertices = []
        while len(vertices) < n_verts:
            line = strip_comments(file.readline())
            if not line:
                continue
            vertices.append(tuple(map(float, line.split())))

        # Read the faces
        faces = []
        while len(faces) < n_faces:
            line = strip_comments(file.readline())
            if not line:
                continue
            face_data = list(map(int, line.split()))
            faces.append(face_data[1:])  # face_data[0] is the number of vertices in the face
    vertices_np = np.array(vertices, dtype=np.float64) #(v,3)
    faces_np = np.array(faces, dtype=np.int32) #(f,3)
    return vertices_np, faces_np   

def save_to_obj(he_trimesh):
    V = he_trimesh.V
    F = he_trimesh.F
    # Filter out unreferenced faces
    referenced_faces = np.delete(F, he_trimesh.unreferenced_faces, axis=0)
    out_path = os.path.join(OUT_DIR, he_trimesh.model_name + '.obj')
    with open(out_path, 'w') as obj_file:
        # Write vertices
        for v in V:
            obj_file.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Write referenced faces (OBJ uses 1-based indexing)
        for face in referenced_faces:
            obj_file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    print(f'saved to {out_path}')