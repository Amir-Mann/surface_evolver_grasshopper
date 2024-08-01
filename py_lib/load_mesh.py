

from collections import defaultdict


NORMAL_DIRECTION = -1


def get_mesh_data(mesh, holes):
    
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

    fixed_vertices = [0] * len(mesh.Vertices)
    fixed_faces = []

    for i in range(mesh.Faces.Count):
        face = mesh.Faces[i]
        face_vertices = [vertex_i_to_unique_id[face.A], vertex_i_to_unique_id[face.B], vertex_i_to_unique_id[face.C]]

        face_edges = []
        for j in range(len(face_vertices)):
            edge = (face_vertices[j], face_vertices[(j + 1) % len(face_vertices)])
            
            # Ensure the edge is ordered consistently
            if (edge[0], edge[1]) in edge_indices.keys():
                index = edge_indices[(edge[0], edge[1])]
                face_edges.append(index)
                fixed_edges[index] = 0

            elif (edge[1], edge[0]) in edge_indices.keys():
                index = edge_indices[(edge[1], edge[0])]
                face_edges.append(-index)
                fixed_edges[index] = 0
                    
            else:
                edge_indices[edge] = edge_index
                face_edges.append(edge_index)
                edge_index += 1 
         
        
        faces.append(face_edges)


    # Identify fixed vertices
    for edge in edge_indices.keys():
        if fixed_edges[edge_indices[edge]] == 1:
            fixed_vertices[edge[0]] = 1
            fixed_vertices[edge[1]] = 1


    # Create fixed faces
    edges_map = defaultdict(tuple)
    for edge in edge_indices:
        if fixed_edges[edge_indices[edge]] == 1 and len(edge) == 2:
            edges_map[edge[0]] = edge
    
    visited_edges = [0] * (len(edge_indices) + 1)


    for edge in edge_indices:
        i = edge_indices[edge]
        if visited_edges[i] == 0 and fixed_edges[i] == 1:
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
                next_edge_index = edge_indices[next_edge]
                face_edges.append(next_edge_index)
                visited_edges[next_edge_index] = 1 
                cur_edge = next_edge
                next_vertex = next_edge[1]
                
            fixed_faces.append(face_edges)


    # Get more boundary conditions
    for holes_mesh in holes:
        for i in range(holes_mesh.Faces.Count):
            face = holes_mesh.Faces[i]
            print(holes_mesh.Vertices.Point3dAt(face.A))
            print(vertex_indices[holes_mesh.Vertices.Point3dAt(face.A)])
            print(holes_mesh.Vertices.Point3dAt(face.B))
            print(vertex_indices[holes_mesh.Vertices.Point3dAt(face.B)])
            print(vertex_indices[holes_mesh.Vertices.Point3dAt(face.C)])

            v1 = vertex_indices[holes_mesh.Vertices.Point3dAt(face.A)]
            v2 = vertex_indices[holes_mesh.Vertices.Point3dAt(face.B)]
            v3 = vertex_indices[holes_mesh.Vertices.Point3dAt(face.C)]

            fixed_vertices[v1] = 1
            fixed_vertices[v2] = 1
            fixed_vertices[v3] = 1

            face_vertices = [v1, v2, v3]
            face_edges = []

            for j in range(len(face_vertices)):
                edge = (face_vertices[j], face_vertices[(j + 1) % len(face_vertices)])

                # Ensure the edge is ordered consistently
                if edge in edge_indices.keys():
                    face_edges.append(edge_indices[edge])

                else:
                    edge = (edge[1], edge[0])
                    face_edges.append(-edge_indices[edge])

                fixed_edges[edge_indices[edge]] = 1

            if face_edges not in faces:
                face_edges = [face_edges[1], face_edges[2], face_edges[0]]

            if face_edges not in faces:
                face_edges = [face_edges[1], face_edges[2], face_edges[0]]

            if face_edges not in faces:
                print("error face not found")

            else:
                faces.remove(face_edges)
                fixed_faces.append(face_edges)

    return vertex_indices, fixed_vertices, edge_indices, fixed_edges, faces, fixed_faces


def parse_mesh(mesh, holes):
    if mesh is None:
        return {}, [], {}, [], [], []
    
    return get_mesh_data(mesh, holes)


def get_mesh_topology_for_fe(arguments):
    vertices, fixed_vertices, edges, fixed_edges, faces, fixed_faces = parse_mesh(arguments["input_mesh"], arguments["input_boundary_conditions"])
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

 