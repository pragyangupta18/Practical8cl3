# Fuzzy set operations and fuzzy relations

def fuzzy_union(A, B):
    return {x: max(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_intersection(A, B):
    return {x: min(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_complement(A):
    return {x: 1 - A[x] for x in A}

def fuzzy_difference(A, B):
    return {x: min(A.get(x, 0), 1 - B.get(x, 0)) for x in set(A) | set(B)}

def cartesian_product(A, B):
    return {(a, b): min(A[a], B[b]) for a in A for b in B}

def max_min_composition(R1, R2):
    result = {}
    xs = set(i for i, _ in R1)
    zs = set(k for _, k in R2)

    for x in xs:
        for z in zs:
            min_values = []
            for y in set(j for _, j in R1):
                val1 = R1.get((x, y), 0)
                val2 = R2.get((y, z), 0)
                min_values.append(min(val1, val2))
            result[(x, z)] = max(min_values) if min_values else 0
    return result

# Example fuzzy sets
A = {'x1': 0.2, 'x2': 0.7, 'x3': 1.0}
B = {'x2': 0.5, 'x3': 0.9, 'x4': 0.4}

# Set operations
print("Union:", fuzzy_union(A, B))
print("Intersection:", fuzzy_intersection(A, B))
print("Complement of A:", fuzzy_complement(A))
print("Difference A - B:", fuzzy_difference(A, B))

# Create fuzzy relations
C = {'y1': 0.8, 'y2': 0.3}
R1 = cartesian_product(A, C)
R2 = cartesian_product(C, B)

print("Fuzzy Relation R1 (A x C):", R1)
print("Fuzzy Relation R2 (C x B):", R2)

# Max-Min Composition of R1 and R2
composed = max_min_composition(R1, R2)
print("Max-Min Composition (R1 o R2):", composed)
