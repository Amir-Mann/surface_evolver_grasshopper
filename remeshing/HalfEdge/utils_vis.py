import numpy as np

import open3d as o3d
import open3d.visualization.gui as gui # type: ignore
from open3d.geometry import TriangleMesh # type: ignore

def get_clean_o3d_trimesh(he_trimesh):
        vertices = o3d.utility.Vector3dVector(he_trimesh.V)
        faces = o3d.utility.Vector3iVector(he_trimesh.F)
        triangle_mesh = TriangleMesh(vertices, faces)
        triangle_mesh.remove_triangles_by_index(he_trimesh.unreferenced_triangles)
        triangle_mesh.remove_vertices_by_index(he_trimesh.unreferenced_vertices)
        return triangle_mesh
   
def vis_he_trimesh(he_trimesh, wireframe=True, v_labels=True, e_labels=True, f_labels=True):
    referenced_vertices_d, referenced_faces_d = dict(), dict()
    faces_midpoints_d = dict()
    # popuulate vertices_d only with the referenced vertices
    for v_idx in range(0, len(he_trimesh.V)):
        if v_idx in he_trimesh.unreferenced_vertices:
            continue
        referenced_vertices_d[v_idx] = he_trimesh.V[v_idx]
    # populate faces_d only with the referenced triangles
    for t_idx in range(0, len(he_trimesh.F)):
        if t_idx in he_trimesh.unreferenced_triangles:
            continue
        referenced_faces_d[t_idx] = he_trimesh.F[t_idx]  # (3,)  int32
        faces_midpoints_d[t_idx] = he_trimesh.get_face_midpoint(t_idx)
    # vertices_np = np.array(list(referenced_vertices_d.values())).astype(np.double) # (v',3)
    # faces_np = np.array(list(referenced_faces_d.values())).astype(np.int32) #(f',3) 
    # vertices = o3d.utility.Vector3dVector(vertices_np)
    # faces = o3d.utility.Vector3iVector(faces_np)
    # mesh = o3d.geometry.TriangleMesh(vertices, faces)
    mesh = get_clean_o3d_trimesh(he_trimesh)
    app = gui.Application.instance
    app.initialize()

    mesh.compute_vertex_normals()
    
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = False
    vis.add_geometry("Mesh", mesh)
    if wireframe:
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        vis.add_geometry("Line", line_set)
    if v_labels: # iterate over referenced_vertices_d 
        for idx, vertex in referenced_vertices_d.items():
            vis.add_3d_label(vertex-[0.0,0.05,0.0], "v{}".format(idx))
    if e_labels:
        edge_midpoints = he_trimesh.get_edges_midpoints() # dict {halfedge_idx: midpoint}
        # edge_midpoints_np = np.array(list(edge_midpoints.values())).astype(np.double)
        # cloud = o3d.geometry.PointCloud(); cloud.points = o3d.utility.Vector3dVector(edge_midpoints_np)
        # vis.add_geometry("Edge Cloud", cloud)
        for he_idx, e_midpoint in edge_midpoints.items():
            he_face_idx = he_trimesh.half_edges[he_idx].face
            twin_idx = he_trimesh.get_twin_index(he_idx)
            twin_face_idx = he_trimesh.half_edges[twin_idx].face
            he_string = "{},{}".format(he_idx, twin_idx) if he_face_idx < twin_face_idx else "{},{}".format(twin_idx, he_idx)
            vis.add_3d_label(e_midpoint-[0.0,0.05,0.0], he_string)
        
    if f_labels:
        face_midpoints = he_trimesh.get_triangles_midpoints()
        # face_midpoints_np = np.array(list(face_midpoints.values())).astype(np.double)
        # cloud = o3d.geometry.PointCloud(); cloud.points = o3d.utility.Vector3dVector(face_midpoints_np)
        # vis.add_geometry("Face Cloud", cloud)
        for idx, midpoint in face_midpoints.items():
            vis.add_3d_label(midpoint-[0.0,0.05,0.0], "t{}".format(idx))
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()

