

import re


SCALE_FACTOR = 1
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
        elif re.findall(r"^-?\d+(\.\d+)?(e(-|\+)\d+)?$", item):
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
    full_faces = []
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
                items = get_line_items(line)
                if "fixed" in items:
                    edge1, edge2, edge3 = items[1:4]
                    assert edges_to_verts[edge1][1] == edges_to_verts[edge2][0]
                    assert edges_to_verts[edge2][1] == edges_to_verts[edge3][0]
                    assert edges_to_verts[edge3][1] == edges_to_verts[edge1][0]
                    full_faces.append((verts_id_to_index[edges_to_verts[edge1][0]],
                                       verts_id_to_index[edges_to_verts[edge2][0]],
                                       verts_id_to_index[edges_to_verts[edge3][0]]))
                edge1, edge2, edge3 = items[1:4]
                assert edges_to_verts[edge1][1] == edges_to_verts[edge2][0]
                assert edges_to_verts[edge2][1] == edges_to_verts[edge3][0]
                assert edges_to_verts[edge3][1] == edges_to_verts[edge1][0]
                faces.append((verts_id_to_index[edges_to_verts[edge1][0]], 
                              verts_id_to_index[edges_to_verts[edge2][0]], 
                              verts_id_to_index[edges_to_verts[edge3][0]]))
                full_faces.append((verts_id_to_index[edges_to_verts[edge1][0]],
                                   verts_id_to_index[edges_to_verts[edge2][0]],
                                   verts_id_to_index[edges_to_verts[edge3][0]]))

    return (verts, faces, full_faces)


def reconstruct_mesh(arguments, results_text):
    verts, faces, full_faces = create_mesh(results_text)
    arguments['result_mesh']['verts'], arguments['result_mesh']['faces'] = verts, faces
    arguments['result_fixed']['verts'], arguments['result_fixed']['faces'] = verts, full_faces
