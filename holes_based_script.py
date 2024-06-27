import rhinoscriptsyntax as rs
import Rhino.Geometry as rg
from collections import defaultdict
import os
import pathlib
import Rhino as rc
from Rhino.Geometry import Mesh
import re
import subprocess

NORMAL_DIRECTION=-1

TEMP_FE_PATH = r"C:\Evolver\temp_fe_file_for_grasshopper_script.fe"
TEMP_DMP_PATH = r"C:\Evolver\temp_fe_file_for_grasshopper_script.dmp"
SE_PATH = r"C:\Evolver\evolver.exe"
LEFT_QURLY_BRACKET = r"{"
RIGHT_QURLY_BRACKET = r"}"
INTER_ACTIVE = False

SCALE_FACTOR = 1
VERTCIES_START = "vertices        /*  coordinates  */    \n"
EDGES_START = "edges  "
FACETS_START = "faces    /* edge loop */      "


## Construct geometry ##


def get_mesh_data(mesh):
    
    # Extract vertices
    
    # Create a dictionary to map vertex positions to their indices. Vertices indexing begins from 0
    vertex_indices = {}
    for i in range(mesh.Vertices.Count):
        vertex = mesh.Vertices.Point3dAt(i)
        if vertex not in vertex_indices:
            vertex_indices[vertex] = i

    # Create a map to make unique vertices ids
    vertex_i_to_unique_id = {}
    for i in range(mesh.Vertices.Count):
        vertex_i_to_unique_id[i] = vertex_indices[mesh.Vertices.Point3dAt(i)]

      
    # Extract edges
    
    # Create a dictionary to map edge tuples to their indices
    edge_indices = {}  

    # Extract faces and identify fixed edges. Edges indexing begins from 1
    faces = []
    fixed_edges = [1] * (mesh.Faces.Count *3)
    edge_index = 1

    for i in range(mesh.Faces.Count):
        face = mesh.Faces[i]
        face_vertices = [vertex_i_to_unique_id[face.A], vertex_i_to_unique_id[face.B], vertex_i_to_unique_id[face.C]]

        face_edges = []
        for j in range(len(face_vertices)):
            edge = (face_vertices[j], face_vertices[(j + 1) % len(face_vertices)])
            
            # Ensure the edge is ordered consistently
            if (edge[1], edge[0]) in edge_indices.keys():
                index = edge_indices[(edge[1], edge[0])]
                face_edges.append(-index)
                fixed_edges[index] = 0
                    
            else:
                edge_indices[edge] = edge_index
                face_edges.append(edge_index)
                edge_index += 1 
         
        
        faces.append(face_edges)

    # Identify fixed vertices
    fixed_vertices = [0] * len(mesh.Vertices)
    for edge in edge_indices.keys():
        if fixed_edges[edge_indices[edge]] == 1:
            fixed_vertices[edge[0]] = 1
            fixed_vertices[edge[1]] = 1        


    # Create fixed faces
    edges_map = defaultdict(list)
    for edge in edge_indices:
        if fixed_edges[edge_indices[edge]] == 1:
            edges_map[edge[0]] = edge
    
    visited_edges = [0] * (len(edge_indices) + 1)
    fixed_faces = []

    for edge in edge_indices:
        i = edge_indices[edge]
        if visited_edges[i] == 0 and fixed_edges[i] == 1:
            start_edge = edge
            cur_edge = start_edge
            face_edges = [i]
            visited_edges[i] = 1
            next_vertex = cur_edge[1]
            
            while 1:
                next_edge = edges_map[next_vertex]   
                if next_edge == start_edge:
                    break
                
                next_edge_index = edge_indices[next_edge]    
                face_edges.append(next_edge_index)
                visited_edges[next_edge_index] = 1 
                cur_edge = next_edge
                next_vertex = next_edge[1]
                
            fixed_faces.append(face_edges)
    
    return vertex_indices, fixed_vertices, edge_indices, fixed_edges, faces, fixed_faces

def parse_mesh(mesh):
    if mesh is None:
        return {}, [], {}, [], [], []
    
    return get_mesh_data(mesh)

def get_mesh_topology_for_fe():
    vertices, fixed_vertices, edges, fixed_edges, faces, fixed_faces = parse_mesh(input_mesh)
    gemotry_text = ""
    # Write vertices
    gemotry_text += 'vertices\n'
    init = True
    for v, i in vertices.items():
        if init:
            init = False
            max_X, max_Y, max_Z, min_X, min_Y, min_Z = v.X, v.Y, v.Z, v.X, v.Y, v.Z
        if v.X > max_X:
            max_X = v.X
        if v.Y > max_Y:
            max_Y = v.Y
        if v.Z > max_Z:
            max_Z = v.Z
        if v.X < min_X:
            min_X = v.X
        if v.Y < min_Y:
            min_Y = v.Y
        if v.Z < min_Z:
            min_Z = v.Z
    for v, i in vertices.items():
        if fixed_vertices[i] == 0:
            gemotry_text += f"{i+1} {v.X + max_X:.2f} {v.Y:.2f} {v.Z:.2f}\n"
        else:
            gemotry_text += f"{i+1} {v.X + max_X:.2f} {v.Y:.2f} {v.Z:.2f} fixed\n"
        
    # Write edges
    gemotry_text += '\nedges\n'
    for edge, index in edges.items():
        if fixed_edges[index] == 0:
            gemotry_text += f"{index}\t{edge[0]+1} {edge[1]+1}\n"
        else:
           gemotry_text += f"{index}\t{edge[0]+1} {edge[1]+1} fixed\n"

    # Write faces
    gemotry_text += '\nfaces\n'
    face_index = 1
    for face in faces:
        single_line = " ".join(map(str, face))
        gemotry_text += f"{face_index}\t{single_line}\n"
        face_index+=1
    for face in fixed_faces:
        face = [-NORMAL_DIRECTION * e for e in face]  
        single_line = " ".join(map(str, face))
        gemotry_text += f"{face_index}  {single_line} fixed\n"  # With a minus sign, to indicate opposite direction for normals to be consistent
        face_index+=1  

    # Write bodies
    gemotry_text += '\nbodies\n'
    gemotry_text += '1\t'
    face_index = 1
    for face in faces:
        gemotry_text += f"{NORMAL_DIRECTION * (face_index)} "
        face_index += 1
    for face in fixed_faces:
        gemotry_text += f"{-NORMAL_DIRECTION * (face_index)} "
        face_index += 1    
    
    gemotry_text += 'density 1 volume 1\n\n'


    gemotry_text += 'read\n'  # additional commands to to Surface Evolver
    gemotry_text += 'N\n'  # Set target volume to actual volume
    gemotry_text += 'set edge color 4 where fixed\n'

    
    volume_estimate = (max_X - min_X) * (max_Y - min_Y) * (max_Z - min_Z)
    return gemotry_text, volume_estimate


## Optimization ##

def get_fe_str():
    fe_file_str, estimated_volume = get_mesh_topology_for_fe()

    fe_file_str += f"read // Take and run SE commands from this file\n"
    fe_file_str += f"G 0; //\n"
    fe_file_str += f"set body target {estimated_volume * 0.5 * volume_factor} where id == 1 // Sets the volume\n"
    if INTER_ACTIVE:
        fe_file_str += f"s // Open graphics window\nq\n"

    fe_file_str += f"optimize_step := {LEFT_QURLY_BRACKET} g; // A general function looking for minimum\n"
    fe_file_str += f"    g {G_INPUT};\n"
    fe_file_str += f"    t .2;\n"
    fe_file_str += f"    V 5;\n"
    fe_file_str += f"    hessian_seek;\n"
    fe_file_str += f"    hessian_seek;\n"
    fe_file_str += f"{RIGHT_QURLY_BRACKET}\n"

    for r in range(R_INPUT):
        fe_file_str += "r; // refine edges \n" if r != 0 else "u; // equiangulation, tries to polish up the triangulation.\n"
        fe_file_str += "optimize_step;\n"

    fe_file_str += f"g {G_INPUT}; // finall settling down\n"
    path_for_fe = TEMP_DMP_PATH.replace("\\", "\\\\")
    fe_file_str += f'dump "{path_for_fe}" // Save results\n'
    if not INTER_ACTIVE:
        fe_file_str += "quit;\n"
        fe_file_str += "q;\n"
    return fe_file_str

def clean_temps():
    if os.path.isfile(TEMP_FE_PATH):
        pathlib.Path.unlink(TEMP_FE_PATH)
    if os.path.isfile(TEMP_DMP_PATH):
        pathlib.Path.unlink(TEMP_DMP_PATH)

 
## Load into grasshopper ##

def send_back_to_grasshopper(results_text):
    global result_mesh
    verts, faces = create_mesh(results_text)
    result_mesh = rs.AddMesh(verts, faces)

def get_line_items(line):
    while "  " in line:
        line = line.replace("  ", " ")
    if line[0] == " ":
        line = line[1:]
    items = line.split(" ")
    returned_items = []
    for item in items:
        if item.isdigit() or (item and item[0] == "-" and item[1:].isdigit()):
            returned_items.append(int(item))
        elif re.findall("^-?\d+(\.\d+)?(e(-|\+)\d+)?$", item):
            returned_items.append(float(item))
        else:
            returned_items.append(item)
    return returned_items

def create_mesh(file_string):
    file_string = file_string.replace("\r\n", "\n").replace("\r", "\n")
    start_index = file_string.find(VERTCIES_START) + len(VERTCIES_START)
    if start_index == len(VERTCIES_START) - 1:
        print("Can't find vertcies start in .dmp file, must be a bug in the script or SE version was changed.")
        return
    
    file_string = file_string[start_index:]
    verts = []
    verts_id_to_index = {}
    edges_to_verts = {}
    faces = []
    mod = "add verts"
    for line in file_string.split("\n"):
        if mod == "add verts":
            if line == EDGES_START:
                mod = "track edges"
            elif len(line) > 3:
                id, x, y, z = get_line_items(line)[:4]
                verts_id_to_index[id] = len(verts)
                verts.append((x * SCALE_FACTOR, y * SCALE_FACTOR, z * SCALE_FACTOR))
        
        elif mod == "track edges":
            if line == FACETS_START:
                mod = "add faces"
            elif len(line) > 3:
                edge, vert1, vert2 = get_line_items(line)[:3]
                edges_to_verts[   edge] = (vert1, vert2)
                edges_to_verts[ - edge] = (vert2, vert1)
        
        elif mod == "add faces":
            if len(line) < 3:
                break
            elif len(line) > 3:
                edge1, edge2, edge3 = get_line_items(line)[1:4]
                assert edges_to_verts[edge1][1] == edges_to_verts[edge2][0]
                assert edges_to_verts[edge2][1] == edges_to_verts[edge3][0]
                assert edges_to_verts[edge3][1] == edges_to_verts[edge1][0]
                faces.append((verts_id_to_index[edges_to_verts[edge1][0]], 
                              verts_id_to_index[edges_to_verts[edge2][0]], 
                              verts_id_to_index[edges_to_verts[edge3][0]]))
    return (verts, faces)


## Execution ##

def run_SE():
    global input_mesh
    fe_file_str = get_fe_str()
    with open(f"{TEMP_FE_PATH}", "w") as temp_fe:
        temp_fe.write(fe_file_str)

    subprocess.run(f"{SE_PATH} {TEMP_FE_PATH}")

    if not os.path.isfile(TEMP_DMP_PATH):
        print("Surface Evolver Failed")
        #clean_temps()
        return

    with open(f"{TEMP_DMP_PATH}", "r") as temp_dmp:
        results_text = temp_dmp.read()
    
    send_back_to_grasshopper(results_text)
    #clean_temps()
    return

if not volume_factor:
    volume_factor = 0.5
if not G_INPUT:
    G_INPUT = 20
if not R_INPUT:
    R_INPUT = 2
run_SE()