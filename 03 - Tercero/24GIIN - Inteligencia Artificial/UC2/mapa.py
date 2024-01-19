""" Knuth's Algorithm X for the exact cover problem, 
    using dicts instead of doubly linked circular lists.
    Written by Ali Assaf

    From http://www.cs.mcgill.ca/~aassaf9/python/algorithm_x.html

    and http://www.cs.mcgill.ca/~aassaf9/python/sudoku.txt

    Python 2 / 3 compatible

    Graph colouring version
"""

from __future__ import print_function

import sys
from itertools import product


def solve(X, Y, solution):
    if not X:
        yield list(solution)
    else:
        c = min(X, key=lambda c: len(X[c]))
        Xc = list(X[c])

        for r in Xc:
            solution.append(r)
            cols = select(X, Y, r)
            for s in solve(X, Y, solution):
                yield s
            deselect(X, Y, r, cols)
            solution.pop()


def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols


def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)


# Invert subset collection
def exact_cover(X, Y):
    newX = dict((j, set()) for j in X)
    for i, row in Y.items():
        for j in row:
            newX[j].add(i)
    return newX


def colour_map(nodes, edges, ncolours=4):
    colours = range(ncolours)

    # The edges that meet each node
    node_edges = dict((n, set()) for n in nodes)
    for e in edges:
        n0, n1 = e
        node_edges[n0].add(e)
        node_edges[n1].add(e)

    for n in nodes:
        node_edges[n] = list(node_edges[n])

    # Set to cover
    coloured_edges = list(product(colours, edges))
    X = nodes + coloured_edges

    # Subsets to cover X with
    Y = dict()
    # Primary rows
    for n in nodes:
        ne = node_edges[n]
        for c in colours:
            Y[(n, c)] = [n] + [(c, e) for e in ne]

    # Dummy rows
    for i, ce in enumerate(coloured_edges):
        Y[i] = [ce]

    X = exact_cover(X, Y)

    # Set first two nodes
    partial = [(nodes[0], 0), (nodes[1], 1)]
    for s in partial:
        select(X, Y, s)

    for s in solve(X, Y, []):
        s = partial + [u for u in s if not isinstance(u, int)]
        s.sort()
        yield s


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

input_polygons = [
    (0, None),
    (1, None),
    (2, None),
    (3, None),
    (4, None),
    (5, None),
    (6, None),
    (7, None),
    (8, None),
]

# graph array of neighbors ids
adjacent_polygons_graph = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 0),
    (2, 0),
    (2, 3),
    (3, 0),
    (3, 2),
    (3, 6),
    (3, 8),
    (4, 0),
    (4, 5),
    (4, 6),
    (5, 0),
    (5, 4),
    (6, 3),
    (6, 4),
    (6, 7),
    (7, 6),
    (8, 3),
]

# Extract the nodes list
nodes = [t[0] for t in input_polygons]

# Get an iterator of all solutions with 3 colours

all_solutions = colour_map(nodes, adjacent_polygons_graph, ncolours=3)

# Print all solutions
# for count, solution in enumerate(all_solutions, start=1):
#     print("%2d: %s" % (count, solution))

output_polygons = next(all_solutions)
print(output_polygons)

# Create a Graphviz DOT file from graph data
colors = {None: "white", 0: "red", 1: "yellow", 2: "green", 3: "blue"}


def graph_to_dot(nodes, edges, outfile=sys.stdout):
    outfile.write("strict graph test{\n")
    outfile.write("    node [style=filled];\n")

    for n, v in nodes:
        outfile.write("    {} [fillcolor={}];\n".format(n, colors[v]))
    outfile.write("\n")

    for u, v in edges:
        outfile.write("    {} -- {};\n".format(u, v))
    outfile.write("}\n")


# Produce a DOT file of the colored graph
with open("graph.dot", "w") as f:
    graph_to_dot(output_polygons, adjacent_polygons_graph, f)
