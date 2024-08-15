import numpy as np
import json
import os 

import open3d as o3d
from half_edge import HalfEdgeModel
from open3d.geometry import HalfEdge, TriangleMesh, HalfEdgeTriangleMesh # type: ignore


SAMPLES_PATH = "/home/ehud/technion/surf/remeshing/samples/"

def load_np(filename):
    if filename.endswith('.json'):
        return load_np_from_json(filename)
    elif filename.endswith('.obj'):
        return load_np_from_obj(filename)

def load_np_from_json(json_filename):
    json_path = os.path.join(SAMPLES_PATH, json_filename)
    with open(json_path, 'r') as file:
        data = json.load(file)
    vertices = np.array(data['verts'], dtype=np.double) # shape (#vertices, 3)
    triangles = np.array(data['faces'], dtype=np.int32) # shape (#triangles, 3)
    
    return vertices, triangles

def load_np_from_obj(obj_filename):
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

def load_halfedge_model_from_obj(file_path):
    vertices_np, faces_np = load_np_from_obj(file_path) # (v,3) float64, (f,3) int32
    vertices = o3d.utility.Vector3dVector(vertices_np)
    faces = o3d.utility.Vector3iVector(faces_np)
    halfedge_model = HalfEdgeModel(vertices, faces)
    return halfedge_model

def load_trimesh_from_json(json_name):
    vertices_np, triangles_np = load_np_from_json(json_name=json_name)
    vertices = o3d.utility.Vector3dVector(vertices_np)
    triangles = o3d.utility.Vector3iVector(triangles_np)
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)
    return mesh

def load_halfedge_trimesh_from_json(json_name):
    trimesh = load_trimesh_from_json(json_name)
    halfedge_trimesh = HalfEdgeTriangleMesh.create_from_triangle_mesh(trimesh)
    return halfedge_trimesh

def load_halfedge_model_from_json(json_name):
    vertices, triangles = load_np_from_json(json_name)
    halfedge_model = HalfEdgeModel(vertices, triangles)
    return halfedge_model

def trimesh_from_halfedge_model(halfedge_model):
    # TODO this ignores unreferenced vertices and triangles
    vertices = halfedge_model.vertices # vector3d
    triangles = halfedge_model.triangles
    trimesh = o3d.geometry.TriangleMesh(vertices, triangles)
    return trimesh