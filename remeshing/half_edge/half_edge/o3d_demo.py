import open3d as o3d
import numpy as np
import json
from open3d.geometry import HalfEdge, TriangleMesh, HalfEdgeTriangleMesh # type: ignore
print(o3d.__version__)
from half_edge import HalfEdgeModel



def load_mesh_from_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Extract vertices and faces from JSON data
    vertices = np.array(data['verts'], dtype=np.double) # shape (#vertices, 3)
    triangles = np.array(data['faces'], dtype=np.int32) # shape (#triangles, 3)
    
    return vertices, triangles

samples_path = "/home/ehud/technion/surface_evolver_grasshopper/remeshing/samples/"
hex_grid_path = samples_path + "hex_grid.json"
vertices_np, triangles_np = load_mesh_from_json(hex_grid_path)
vertices = o3d.utility.Vector3dVector(vertices_np)
triangles = o3d.utility.Vector3iVector(triangles_np)

# mesh = HalfEdgeModel(vertices, triangles)
mesh = TriangleMesh(vertices, triangles)
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, mesh_show_wireframe=True)

vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the mesh to the visualizer
vis.add_geometry(mesh)

# Get the vertices of the mesh
vertices = np.asarray(mesh.vertices, dtype=np.float32)
text_3d_geos = []
# Add vertex labels
# for i, vertex in enumerate(vertices):
    # text = f'{i}'  # Vertex index
    # print(f'Vertex {i}: {vertex}, type: {type(vertex)}')
    # text_3d = o3d.visualization.gui.Label3D(text, vertex[:,np.newaxis])
    # vis.add_geometry(text_3d)

# Run the visualizer
vis.run()
vis.destroy_window()

