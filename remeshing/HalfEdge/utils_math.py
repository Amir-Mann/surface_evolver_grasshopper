import numpy as np
def angle(vertices):
    # vertices is a (2,3) ndarray of two vectors in R^3,
    # returns the angle between the two vectors
    norm = np.linalg.norm(vertices, axis=1)
    ip = np.dot(vertices[0] / norm[0], vertices[1] / norm[1])
    ip = np.clip(ip, -1, 1)
    return np.arccos(ip)


### Triangle class

def is_collinear(vertices):
    # vertices is (3,3) ndarray, each row a vertex,
    # checks whether the points lie on the same line or close enough
    vectors = vertices[1:] - vertices[0] # (2,3) ndarray
    cross = np.cross(vectors[0], vectors[1])
    return np.allclose(cross, 0)

def get_normal(vertices):
    # vertices is (3,3) float64 ndarray, each row a vertex,
    # returns the normal of the triangle
    v0, v1, v2 = vertices
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    normal = np.cross(v0v1, v0v2)
    return normal / np.linalg.norm(normal)

def get_vectors(vertices):
    # vertices is (3,3) float64 ndarray, each row a vertex,
    # returns (2,3) the vectors of the triangle
    v0, v1, v2 = vertices
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    return np.vstack([v0v1, v0v2]) 

def get_lengths(vertices):
    # vertices is (3,3) float64 ndarray, each row a vertex,
    # returns (3,) float64 the lengths of the edges
    a = np.linalg.norm(vertices[1] - vertices[0])
    b = np.linalg.norm(vertices[2] - vertices[1])
    c = np.linalg.norm(vertices[0] - vertices[2])
    return np.array([a,b,c])

def get_area(vertices):
    # vertices is (3,3) float64 ndarray, each row a vertex,
    # returns the area of the triangle, using heron's formula
    lengths = get_lengths(vertices) # (3,)
    s = np.sum(lengths) / 2
    # fix for numerical stability
    mask = np.isclose(s, lengths) # fix rounding issue.
    aux = np.where(mask, 0., s-lengths) # TODO check
    return np.sqrt(s*np.prod(aux))

def get_compactness(vertices):
    # vertices is (3,3) float64 ndarray, each row a vertex,
    # returns how much the triangle is close to an equilateral triangle
    # let L be perimeter, A the area,
    # the isoperimetric inequality promises that L^2 >= 4 * pi * A
    # TODO not sure about formula, look at
    # maybe answer in https://math.berkeley.edu/~wu/HSI1.pdf
    # if the value is 1, the triangle is equilateral
    # if the value is 0, the triangle is degenerate
    lengths = get_lengths(vertices) # (3,)
    triangle_area = get_area(vertices) # float64, for isoceles is equal to (sqrt(3)/4) * side**2
    numerator = 4 * np.sqrt(3) * triangle_area
    denom = np.linalg.norm(lengths)**2
    return numerator / denom

def get_normal(vertices, normalize=True):
    # vertices is (3,3) float64 ndarray, each row a vertex,
    # returns the normal (3,) of the triangle
    vectors = get_vectors(vertices) # (2,3) ndarray
    normal = np.cross(vectors[0], vectors[1])
    if normalize:
        return normal / np.linalg.norm(normal)
    return normal

def get_cross_product_norm(vertices):
    # vertices is (3,3) float64 ndarray, each row a vertex,
    # returns the norm of the cross product of the vectors
    cross_prod = get_normal(vertices, normalize=False)
    return np.linalg.norm(cross_prod)