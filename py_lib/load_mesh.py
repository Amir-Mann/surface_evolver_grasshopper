

from collections import defaultdict


NORMAL_DIRECTION = -1
LARGE_NUM = 10000000


class Context:
    def __init__(self):
        self.vertex_indices = {}
        self.fixed_vertices = []
        self.edge_indices = {}
        self.fixed_edges = []
        self.faces = []
        self.fixed_faces = []
        self.bodies = []
        

def get_mesh_data(mesh, holes, c):
    main_mesh = c is None
    print("Processing mesh, main_mesh:", main_mesh)
    if main_mesh:
        c = Context()
    
    # Extract vertices
    # Create a dictionary to map vertex positions to their indices. Vertices indexing begins from 0
    start_index = len(c.vertex_indices)
    vertex_index = start_index + 1  # Start from 1 for Surface Evolver compatibility
    for i in range(mesh.Vertices.Count):
        vertex = mesh.Vertices.Point3dAt(i)
        if vertex not in c.vertex_indices:
            c.vertex_indices[vertex] = vertex_index
            vertex_index += 1

    # Create a map to make unique vertices ids
    vertex_i_to_unique_id = {}
    for i in range(mesh.Vertices.Count):
        vertex_i_to_unique_id[i] = c.vertex_indices[mesh.Vertices.Point3dAt(i)]  
    
    # Extract edges
    # Extract faces and identify fixed edges. Edges indexing begins from 1
    c.fixed_edges += [1] * (mesh.Faces.Count *3)
    
    start_edge_index = len(c.edge_indices) + 1
    edge_index = start_edge_index
    c.fixed_vertices += [0] * len(mesh.Vertices)

    start_face_index = len(c.faces) + 1
    c.bodies.append([])
    for i in range(mesh.Faces.Count):
        face = mesh.Faces[i]
        face_vertices = [vertex_i_to_unique_id[face.A], vertex_i_to_unique_id[face.B], vertex_i_to_unique_id[face.C]]

        face_edges = []
        for j in range(len(face_vertices)):
            edge = (face_vertices[j], face_vertices[(j + 1) % len(face_vertices)])
            
            # Ensure the edge is ordered consistently
            if (edge[0], edge[1]) in c.edge_indices.keys():
                index = c.edge_indices[(edge[0], edge[1])]
                face_edges.append(index)
                c.fixed_edges[index] = 0

            elif (edge[1], edge[0]) in c.edge_indices.keys():
                index = c.edge_indices[(edge[1], edge[0])]
                face_edges.append(-index)
                c.fixed_edges[index] = 0
                    
            else:
                c.edge_indices[edge] = edge_index
                face_edges.append(edge_index)
                edge_index += 1
         
        c.faces.append(face_edges)
        c.bodies[-1].append(NORMAL_DIRECTION * (start_face_index + i))
        if not main_mesh:
            c.bodies[0].append( - NORMAL_DIRECTION * (start_face_index + i))


    # Identify fixed vertices
    for edge in c.edge_indices.keys():
        if c.fixed_edges[c.edge_indices[edge]] == 1:
            c.fixed_vertices[edge[0]] = 1
            c.fixed_vertices[edge[1]] = 1



    # Create fixed faces
    edges_map = defaultdict(tuple)
    for edge in c.edge_indices:
        if c.fixed_edges[c.edge_indices[edge]] == 1 and len(edge) == 2:
            edges_map[edge[0]] = edge
    
    visited_edges = [1] * start_edge_index + [0] * (len(c.edge_indices) + 1 - start_edge_index)

    for edge in c.edge_indices:
        i = c.edge_indices[edge]
        if visited_edges[i] == 0 and c.fixed_edges[i] == 1:
            start_edge = edge
            cur_edge = start_edge
            face_edges = [i]
            visited_edges[i] = 1
            next_vertex = cur_edge[1]
            
            while 1:
                #print(next_vertex)
                if next_vertex not in edges_map.keys():
                    continue

                next_edge = edges_map[next_vertex]
                if next_edge == start_edge:
                    break

                #print(next_edge)
                next_edge_index = c.edge_indices[next_edge]
                face_edges.append(next_edge_index)
                visited_edges[next_edge_index] = 1 
                cur_edge = next_edge
                next_vertex = next_edge[1]
            c.fixed_faces.append(face_edges)
            c.bodies[-1].append( - NORMAL_DIRECTION * (LARGE_NUM + len(c.fixed_faces)))
            if not main_mesh:
                c.bodies[0].append(NORMAL_DIRECTION * (LARGE_NUM + len(c.fixed_faces)))


    # Get more boundary conditions
    for holes_mesh in holes:
        if holes_mesh is None or holes_mesh.Vertices.Count == 0:
            continue
        for i in range(holes_mesh.Faces.Count):
            face = holes_mesh.Faces[i]
            try:
                v1 = c.vertex_indices[holes_mesh.Vertices.Point3dAt(face.A)]
                v2 = c.vertex_indices[holes_mesh.Vertices.Point3dAt(face.B)]
                v3 = c.vertex_indices[holes_mesh.Vertices.Point3dAt(face.C)]
            except KeyError:
                print("Boundary mesh has points not on input mesh, skipping.")
                continue

            c.fixed_vertices[v1] = 1
            c.fixed_vertices[v2] = 1
            c.fixed_vertices[v3] = 1

            face_vertices = [v1, v2, v3]
            face_edges = []

            for j in range(len(face_vertices)):
                edge = (face_vertices[j], face_vertices[(j + 1) % len(face_vertices)])

                # Ensure the edge is ordered consistently
                if edge in c.edge_indices.keys():
                    face_edges.append(c.edge_indices[edge])

                else:
                    edge = (edge[1], edge[0])
                    face_edges.append(-c.edge_indices[edge])

                c.fixed_edges[c.edge_indices[edge]] = 1

            if face_edges not in c.faces:
                face_edges = [face_edges[1], face_edges[2], face_edges[0]]

            if face_edges not in c.faces:
                face_edges = [face_edges[2], face_edges[0], face_edges[1]]

            if face_edges not in c.faces:
                print("error face not found")
            else:
                c.faces.remove(face_edges)
                c.fixed_faces.append(face_edges)

    edge_length_sum_none_fixed = 0
    edge_count_none_fixed = 0
    for edge in c.edge_indices.keys():
        if c.fixed_edges[c.edge_indices[edge]] != 1:
            v1 = edge[0]
            v2 = edge[1]
            edge_length_sum_none_fixed += ((mesh.Vertices.Point3dAt(v1) - mesh.Vertices.Point3dAt(v2)).Length)
            edge_count_none_fixed += 1
    
    return c, edge_length_sum_none_fixed/edge_count_none_fixed


def parse_mesh(mesh, holes, c=None):
    if mesh is None:
        return {}, [], {}, [], [], []
    
    return get_mesh_data(mesh, holes, c=c)


def get_mesh_topology_for_fe(arguments):
    context, edge_length_avg_none_fixed = parse_mesh(arguments["input_mesh"], arguments["input_boundary_conditions"])
    for mesh in arguments["inner_meshes"]:
        context, _ = get_mesh_data(mesh, [], c=context)
    gemotry_text = ""
    # Write vertices
    gemotry_text += 'vertices\n'
    init = True
    for v, i in context.vertex_indices.items():
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
    for v, i in context.vertex_indices.items():
        if context.fixed_vertices[i] == 0:
            gemotry_text += f"{i+1} {v.X + (max_X - min_X) * 1.6:.2f} {v.Y:.2f} {v.Z:.2f}\n"
        else:
            gemotry_text += f"{i+1} {v.X + (max_X - min_X) * 1.6:.2f} {v.Y:.2f} {v.Z:.2f} fixed\n"
        
    # Write edges
    gemotry_text += '\nedges\n'
    for edge, index in context.edge_indices.items():
        if context.fixed_edges[index] == 0:
            gemotry_text += f"{index}\t{edge[0]+1} {edge[1]+1}\n"
        else:
            gemotry_text += f"{index}\t{edge[0]+1} {edge[1]+1} fixed\n"

    # Write faces
    gemotry_text += '\nfaces\n'
    face_index = 1
    for face in context.faces:
        single_line = " ".join(map(str, face))
        gemotry_text += f"{face_index}\t{single_line}\n"
        face_index+=1
    for face in context.fixed_faces:
        face = [-NORMAL_DIRECTION * e for e in face]  
        single_line = " ".join(map(str, face))
        gemotry_text += f"{face_index}  {single_line} fixed\n"  # With a minus sign, to indicate opposite direction for normals to be consistent
        face_index+=1  

    # Write bodies
    gemotry_text += '\nbodies\n'
    body_index = 1
    
    def fix_body_index(index): 
        # This is some discussting code to fix body indexes, all fixed body faces are added LARGE_NUM
        # To fix them we replace LARGE_NUM with the number of faces in the context
        if abs(index) < LARGE_NUM:
            return str(index)
        else:
            sign = -1 if index < 0 else 1
            abs_index = abs(index) - LARGE_NUM + len(context.faces)
            return str(sign * abs_index)
    
    for body in context.bodies:
        body_text = " ".join(map(fix_body_index, body))
        gemotry_text += f"{body_index}\t{body_text} "
        gemotry_text += 'density 1 volume 1\n'
        body_index += 1
    gemotry_text += '\n'

    gemotry_text += 'read\n'  # additional commands to to Surface Evolver
    gemotry_text += 'N\n'  # Set target volume to actual volume
    gemotry_text += 'set edge color 4 where fixed\n'

    return gemotry_text, (max_X - min_X), (max_Y - min_Y), (max_Z - min_Z), edge_length_avg_none_fixed

 