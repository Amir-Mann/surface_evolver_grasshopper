

import os
from remesh import Mesh
import unittest

class TestMesh(unittest.TestCase):
    def assertMeshIntegrity(self, mesh):
        for face in mesh.faces:
            self.assertEqual(len(face.verts), 3)
            for vert in face.verts:
                self.assertIn(vert, mesh.verts)
                self.assertIn(face, vert.faces)
        for vert in mesh.verts:
            self.assertGreater(len(vert.faces), 0)
            for face in vert.faces:
                self.assertIn(vert, face.verts)
                self.assertIn(face, mesh.faces)

class TestUnite(TestMesh):
    def test_unite_non_boundary(self):
        mesh = Mesh(os.path.join(".", "samples", "hex_grid.json"))
        v1, v2 = mesh.verts[5], mesh.verts[10]
        f1, f2 = mesh.faces[8], mesh.faces[9]
        v1.unite(v2, mesh)
        self.assertEqual(len(mesh.faces), 16)
        self.assertNotIn(v2, mesh.verts)
        self.assertIn(v1, mesh.verts)
        self.assertNotIn(f1, mesh.faces)
        self.assertNotIn(f2, mesh.faces)
        self.assertEqual(len(v1.faces), 8)
        for face in v1.faces:
            self.assertIn(face, mesh.faces)
            self.assertIn(v1, face.verts)
        self.assertMeshIntegrity(mesh)
            
class TestSplitEdge(TestMesh):
    def test_split_edge_non_boundary(self):
        mesh = Mesh(os.path.join(".", "samples", "hex_grid.json"))
        v1, v2 = mesh.verts[5], mesh.verts[10]
        v1.split_edge(v2, mesh)
        self.assertMeshIntegrity(mesh)

class TestEdgeFlip(TestMesh):
    def test_edge_flip_non_boundary(self):
        mesh = Mesh(os.path.join(".", "samples", "hex_grid.json"))
        f1, f2 = mesh.faces[8], mesh.faces[9]
        f1.maybe_edge_flip(f2)
        self.assertMeshIntegrity(mesh)
        
if __name__ == "__main__":
    unittest.main()