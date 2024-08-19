import numpy as np

import open3d as o3d
import open3d.visualization.gui as gui # type: ignore
from open3d.geometry import TriangleMesh # type: ignore

import plotly.graph_objs as go

def vis_he_trimesh(he_trimesh, wireframe=True, v_labels=True, e_labels=True, f_labels=True):
    # plot only referenced vertices, edges and faces
    V = he_trimesh.V
    F = he_trimesh.F
    vertices_d = he_trimesh.get_vertices()
    edge_midpoints_d = he_trimesh.get_edges_midpoints()
    face_midpoints_d = he_trimesh.get_faces_midpoints()

    fig = go.Figure()

    # Plot wireframe of the triangle mesh
    if wireframe:
        for f_idx, f_verts_inds in enumerate(F):
            if f_idx in he_trimesh.unreferenced_faces:
                continue
            i, j, k = f_verts_inds
            fig.add_trace(go.Scatter3d(x=[V[i, 0], V[j, 0], V[k, 0], V[i, 0]],
                                       y=[V[i, 1], V[j, 1], V[k, 1], V[i, 1]],
                                       z=[V[i, 2], V[j, 2], V[k, 2], V[i, 2]],
                                       mode='lines',
                                       line=dict(color='lightblue', width=2)))

    # Plot vertices with labels
    if v_labels:
        for idx, vertex in vertices_d.items():
            fig.add_trace(go.Scatter3d(x=[vertex[0]], y=[vertex[1]], z=[vertex[2]], mode='markers+text',
                                       text=[f'v{idx}'], textposition="top center", marker=dict(size=5, color='red')))

    # Plot edge midpoints with labels
    if e_labels:
        for he_idx, edge_midpoint in edge_midpoints_d.items():
            he_face_idx = he_trimesh.half_edges[he_idx].face
            twin_idx = he_trimesh.half_edges[he_idx].twin
            twin_face_idx = he_trimesh.half_edges[twin_idx].face
            he_string = "{},{}".format(he_idx, twin_idx) if he_face_idx < twin_face_idx else "{},{}".format(twin_idx, he_idx)
            fig.add_trace(go.Scatter3d(x=[edge_midpoint[0]], y=[edge_midpoint[1]], z=[edge_midpoint[2]], mode='markers+text', text=[he_string], textposition="top center", marker=dict(size=5, color='green')))

    # Plot face midpoints with labels
    if f_labels:
        for idx, face_midpoint in face_midpoints_d.items():
            fig.add_trace(go.Scatter3d(x=[face_midpoint[0]], y=[face_midpoint[1]], z=[face_midpoint[2]], mode='markers+text', text=[f"f{idx}"], textposition="top center", marker=dict(size=5, color='blue')))

    camera = dict(
        eye=dict(x=0., y=0., z=1.3),
        up=dict(x=0., y=1., z=0.),
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'),
        showlegend=False,
        scene_camera=camera,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig.show()


def get_clean_o3d_trimesh(he_trimesh):
    vert_copy = np.array(he_trimesh.V, copy=True)
    faces_copy = np.array(he_trimesh.F, copy=True)
    vertices = o3d.utility.Vector3dVector(vert_copy)
    faces = o3d.utility.Vector3iVector(faces_copy)
    triangle_mesh = TriangleMesh(vertices, faces)
    triangle_mesh.remove_triangles_by_index(he_trimesh.unreferenced_faces)
    triangle_mesh.remove_vertices_by_index(he_trimesh.unreferenced_vertices)
    return triangle_mesh
   
def vis_he_trimesh_o3d(he_trimesh, wireframe=True, v_labels=True, e_labels=True, f_labels=True):
    referenced_vertices_d = he_trimesh.get_vertices() # TODO remove

    # create o3d mesh
    mesh = get_clean_o3d_trimesh(he_trimesh)
    app = gui.Application.instance
    app.initialize()
    mesh.compute_vertex_normals()  
    # create o3d visualizer 
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = False
    # add mesh to visualizer
    vis.add_geometry("Mesh", mesh)
    
    if wireframe:
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        vis.add_geometry("Line", line_set)
    if v_labels: # iterate over referenced_vertices_d 
        for idx, vertex in referenced_vertices_d.items():
            vis.add_3d_label(vertex-[0.0,0.05,0.0], "v{}".format(idx))
            # vis.add_3d_label(vertex, "v{}".format(idx))
    if e_labels:
        edge_midpoints = he_trimesh.get_edges_midpoints() # dict {halfedge_idx: midpoint}
        # edge_midpoints_np = np.array(list(edge_midpoints.values())).astype(np.double)
        # cloud = o3d.geometry.PointCloud(); cloud.points = o3d.utility.Vector3dVector(edge_midpoints_np)
        # vis.add_geometry("Edge Cloud", cloud)
        for he_idx, e_midpoint in edge_midpoints.items():
            he_face_idx = he_trimesh.half_edges[he_idx].face
            twin_idx = he_trimesh.half_edges[he_idx].twin
            twin_face_idx = he_trimesh.half_edges[twin_idx].face
            he_string = "{},{}".format(he_idx, twin_idx) if he_face_idx < twin_face_idx else "{},{}".format(twin_idx, he_idx)
            vis.add_3d_label(e_midpoint-[0.0,0.05,0.0], he_string)
            # vis.add_3d_label(e_midpoint, he_string)
        
    if f_labels:
        face_midpoints = he_trimesh.get_faces_midpoints()
        # face_midpoints_np = np.array(list(face_midpoints.values())).astype(np.double)
        # cloud = o3d.geometry.PointCloud(); cloud.points = o3d.utility.Vector3dVector(face_midpoints_np)
        # vis.add_geometry("Face Cloud", cloud)
        for idx, midpoint in face_midpoints.items():
            vis.add_3d_label(midpoint-[0.0,0.05,0.0], "t{}".format(idx))
            # vis.add_3d_label(midpoint, "f{}".format(idx))
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()

