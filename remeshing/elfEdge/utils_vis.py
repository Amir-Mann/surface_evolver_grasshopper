import numpy as np

import open3d as o3d
from half_edge import HalfEdgeModel
import open3d.visualization.gui as gui # type: ignore
import open3d.visualization.rendering as rendering # type: ignore

from utils_load import *


def vis_mesh_simple(mesh, backface=True, wireframe=True):
    if type(mesh) == HalfEdgeModel:
        mesh = trimesh_from_halfedge_model(mesh)
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=backface, mesh_show_wireframe=wireframe)
    
def vis_mesh_low(mesh, vert_labels=True):
    app = gui.Application.instance
    app.initialize()
    if type(mesh) == HalfEdgeModel:
        mesh = trimesh_from_halfedge_model(mesh)
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("Mesh", mesh)
    if vert_labels:
        for idx in range(0, len(mesh.vertices)):
            vis.add_3d_label(mesh.vertices[idx], "{}".format(idx))
    app.add_window(vis)
    app.run()
    
    
def vis_mesh_high(mesh, wireframe=True, vert_labels=True):
    if type(mesh) == HalfEdgeModel:
        mesh = trimesh_from_halfedge_model(mesh)
    app = gui.Application.instance
    app.initialize()

    mesh.compute_vertex_normals()
    
    
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("Mesh", mesh)
    if wireframe:
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        vis.add_geometry("Line", line_set)
    if vert_labels:
        for idx in range(0, len(mesh.vertices)):
            vis.add_3d_label(mesh.vertices[idx], "{}".format(idx))
    
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()
    
def vis_halfedge_model_old(halfedge_model, wireframe=True, vert_labels=True, edge_labels=True, face_labels=True):
    mesh = trimesh_from_halfedge_model(halfedge_model)
    vertices_d, faces_d = dict(), dict()
    faces_midpoints_d = dict()
    # popuulate vertices_d only with the referenced vertices
    for v_idx in range(0, len(halfedge_model.vertices)):
        if v_idx in halfedge_model.unreferenced_vertices:
            continue
        vertices_d[v_idx] = halfedge_model.vertices[v_idx]
    # populate faces_d only with the referenced triangles
    for t_idx in range(0, len(halfedge_model.triangles)):
        if t_idx in halfedge_model.unreferenced_triangles:
            continue
        faces_d[t_idx] = halfedge_model.triangles[t_idx]
        indices = halfedge_model.triangles[t_idx] # (3,)  int32
        vertices_t = halfedge_model.get_vertices_by_indices(indices) # (3,3) float64
        midpoint_t = np.mean(vertices_t, axis=0) # (3,)
        faces_midpoints_d[t_idx] = midpoint_t
    # vertices_np = np.array(list(vertices_d.values())).astype(np.double) # (v',3)
    # faces_np = np.array(list(faces_d.values())).astype(np.int32) #(f',3) 
    # vertices = o3d.utility.Vector3dVector(vertices_np)
    # faces = o3d.utility.Vector3iVector(faces_np)
    # mesh = o3d.geometry.TriangleMesh(vertices, faces)
    mesh = halfedge_model.get_clean_trimesh()
    app = gui.Application.instance
    app.initialize()

    mesh.compute_vertex_normals()
    
    
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = False
    vis.add_geometry("Mesh", mesh)
    if wireframe:
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        vis.add_geometry("Line", line_set)
    if vert_labels: # iterate over vertices_d 
        for idx, vertex in vertices_d.items():
            vis.add_3d_label(vertex-[0.0,0.05,0.0], "v{}".format(idx))
    if edge_labels:
        edge_midpoints = halfedge_model.get_edges_midpoints() # dict {halfedge_idx: midpoint}
        # edge_midpoints_np = np.array(list(edge_midpoints.values())).astype(np.double)
        # cloud = o3d.geometry.PointCloud(); cloud.points = o3d.utility.Vector3dVector(edge_midpoints_np)
        # vis.add_geometry("Cloud", cloud)
        for idx, e_midpoint in edge_midpoints.items():
            vis.add_3d_label(e_midpoint-[0.0,0.05,0.0], "{},{}".format(idx, halfedge_model.get_twin_index(idx)))
        
    if face_labels:
        face_midpoints = halfedge_model.get_triangles_midpoints()
        # face_midpoints_np = np.array(list(face_midpoints.values())).astype(np.double)
        # cloud = o3d.geometry.PointCloud(); cloud.points = o3d.utility.Vector3dVector(face_midpoints_np)
        # vis.add_geometry("Cloud", cloud)
        for idx, midpoint in face_midpoints.items():
            vis.add_3d_label(midpoint-[0.0,0.05,0.0], "t{}".format(idx))
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()
    
def vis_halfedge_model(halfedge_model, wireframe=True, vert_labels=True, edge_labels=True, face_labels=True):
    referenced_vertices_d, referenced_faces_d = dict(), dict()
    faces_midpoints_d = dict()
    # popuulate vertices_d only with the referenced vertices
    for v_idx in range(0, len(halfedge_model.V)):
        if v_idx in halfedge_model.unreferenced_vertices:
            continue
        referenced_vertices_d[v_idx] = halfedge_model.V[v_idx]
    # populate faces_d only with the referenced triangles
    for t_idx in range(0, len(halfedge_model.F)):
        if t_idx in halfedge_model.unreferenced_triangles:
            continue
        referenced_faces_d[t_idx] = halfedge_model.F[t_idx]  # (3,)  int32
        faces_midpoints_d[t_idx] = halfedge_model.get_face_midpoint(t_idx)
    # vertices_np = np.array(list(referenced_vertices_d.values())).astype(np.double) # (v',3)
    # faces_np = np.array(list(referenced_faces_d.values())).astype(np.int32) #(f',3) 
    # vertices = o3d.utility.Vector3dVector(vertices_np)
    # faces = o3d.utility.Vector3iVector(faces_np)
    # mesh = o3d.geometry.TriangleMesh(vertices, faces)
    mesh = halfedge_model.get_clean_trimesh()
    app = gui.Application.instance
    app.initialize()

    mesh.compute_vertex_normals()
    
    
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = False
    vis.add_geometry("Mesh", mesh)
    if wireframe:
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        vis.add_geometry("Line", line_set)
    if vert_labels: # iterate over referenced_vertices_d 
        for idx, vertex in referenced_vertices_d.items():
            vis.add_3d_label(vertex-[0.0,0.05,0.0], "v{}".format(idx))
    if edge_labels:
        edge_midpoints = halfedge_model.get_edges_midpoints() # dict {halfedge_idx: midpoint}
        # edge_midpoints_np = np.array(list(edge_midpoints.values())).astype(np.double)
        # cloud = o3d.geometry.PointCloud(); cloud.points = o3d.utility.Vector3dVector(edge_midpoints_np)
        # vis.add_geometry("Edge Cloud", cloud)
        for he_idx, e_midpoint in edge_midpoints.items():
            he_face_idx = halfedge_model.half_edges[he_idx].face
            twin_idx = halfedge_model.get_twin_index(he_idx)
            twin_face_idx = halfedge_model.half_edges[twin_idx].face
            he_string = "{},{}".format(he_idx, twin_idx) if he_face_idx < twin_face_idx else "{},{}".format(twin_idx, he_idx)
            vis.add_3d_label(e_midpoint-[0.0,0.05,0.0], he_string)
        
    if face_labels:
        face_midpoints = halfedge_model.get_triangles_midpoints()
        # face_midpoints_np = np.array(list(face_midpoints.values())).astype(np.double)
        # cloud = o3d.geometry.PointCloud(); cloud.points = o3d.utility.Vector3dVector(face_midpoints_np)
        # vis.add_geometry("Face Cloud", cloud)
        for idx, midpoint in face_midpoints.items():
            vis.add_3d_label(midpoint-[0.0,0.05,0.0], "t{}".format(idx))
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()

