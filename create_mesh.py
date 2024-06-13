from Rhino.Geometry import Mesh
import rhinoscriptsyntax as rs
import re
import os
import subprocess

SCALE_FACTOR = 10
VERTCIES_START = "vertices        /*  coordinates  */    \n"
EDGES_START = "edges  "
FACETS_START = "faces    /* edge loop */      "

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
        elif re.findall("^-?\d+(\.\d+)?(e-?\d+)?$", item):
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
COMMAND = r"C:\Evolver\evolver.exe C:\Users\user\Documents\7th_semester\fluids\surface_evolver_grasshopper\fe_demos\3ballsrunanddump.fe"

subprocess.run(COMMAND)

FILE_PATH = r"C:\Users\user\Documents\7th_semester\fluids\surface_evolver_grasshopper\fe_demos\3ballsrunanddump.dmp"
file_string = open(FILE_PATH, "r").read()
verts, faces = create_mesh(file_string)
print("\n".join(map(str, verts)))
print("faces")
print("\n".join(map(str, faces)))
a = rs.AddMesh(verts, faces)