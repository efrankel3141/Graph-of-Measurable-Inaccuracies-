import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.io import savemat


class G_mi:

    # -----------------------------------------------------
    # GRAPH INITIALIZER ->
    # [num of n-levels, nodes per level, construction-type]
    # -----------------------------------------------------
    def __init__(self, n, k, construction):
        self.n = n
        self.construction = construction
        self.k = k

    # -----------------------------------------------------
    # ADDITIONAL VARIABLES ->
    # G - the graph, nodes - list of all nodes for G,
    # edges - list of all edges for G, weights - weighted
    # adjacency matrix for G, total_nodes - total # of nodes
    # -----------------------------------------------------
    G = nx.Graph()
    nodes = list(G.nodes)
    edges = list(G.edges)
    weights = [[]]
    # -----------------------------------------------------
    # GET NUM OF N-LEVELS
    # -----------------------------------------------------

    def get_n(self):
        return self.n

    # -----------------------------------------------------
    # GET CONSTRUCTION-TYPE
    # -----------------------------------------------------
    def get_const(self):
        return self.construction

    # -----------------------------------------------------
    # GET NUM OF NODES PER N-LEVEL
    # -----------------------------------------------------
    def get_k(self):
        return self.k

    # -----------------------------------------------------
    # GET LIST OF VERTICES (NODES)
    # -----------------------------------------------------
    def get_vertices(self):
        return self.nodes

    # -----------------------------------------------------
    # GET LIST OF EDGES
    # -----------------------------------------------------
    def get_edges(self):
        return self.edges

    # -----------------------------------------------------
    # GET WEIGHTED ADJ. MATRIX
    # -----------------------------------------------------
    def get_weight_matrix(self):
        return self.weights

    # -----------------------------------------------------
    # GET THE WHOLE GRAPH
    # -----------------------------------------------------
    def get_graph(self):
        return self.G

    # -----------------------------------------------------
    # /////////////////////////////////////////////////////
    # -----------------------------------------------------

    # -----------------------------------------------------
    # GET TOTAL NUM OF NODES
    # -----------------------------------------------------
    def get_total_nodes(self):
        return sum([self.k*2**(i-1) for i in range(1, self.n+1)], 1)

    # -----------------------------------------------------
    # GENERATE ADJ. MATRIX - (based off specific graph)
    # looks at the weight for each edge in the edge set for
    # and appropriately encodes it in a matrix with dim.
    # (\sum_{i=0}^n 2^{n-1})^2 or (total_nodes)^2
    # -----------------------------------------------------
    def populate_weights(self):
        self.weights = [[0 for _ in range(self.get_total_nodes())]
                        for _ in range(self.get_total_nodes())]
        for i in range(len(self.weights)):

            for j in range(len(self.weights[i])):

                if self.G.has_edge(i, j):

                    weight = self.G.get_edge_data(
                        i, j).get("weight", None)

                    self.weights[i][j] = weight

    def display_weights(self):
        matrix = ""
        for row in self.weights:
            matrix += str(row)+"\n"
        return matrix

    def display_sub_adj(self, r):
        sub_matrix = ""
        index = 0
        for row in self.weights:
            if index > self.k*(2**(r)):
                sub_matrix += str(row[self.k*(2**(r))+1:])+"\n"
            index += 1
            # print(index)
        return sub_matrix

    # -----------------------------------------------------
    # CONSTRUCT GRAPH ->
    # [k - starting n-level, [] - empty unless k>1]
    #
    # based on choice of construction, the function builds
    # the appropriate model of G_mi to your specs.
    # -----------------------------------------------------
    def construct_graph(self, m, edges, first, last, step):
        t = self.get_total_nodes() - 1
        self.G.add_node(0)
        self.G.add_nodes_from(list(range(1, t+1)))
        # print(self.G.nodes)
        # print(edges)

        if m == 1:
            print("REACHED-1")
            for i in range(1, t+1):
                edges.append((0, i, 1))
            return self.construct_graph(m+1, edges, first, last, step)

        elif m == 2:
            print("REACHED-2")
            for i in range(k+2*(0)+1, t, 2):
                edges.append((i, i+1, 1))
            return self.construct_graph(m+1, edges, first, last, step)

        elif m <= self.n:
            print(f"REACHED-{m}")

            old_first = first
            old_last = last
            old_step = step

            print()
            step = ((old_first + 2**(m-2)) - (old_first + old_step))
            print(f"new step is: {step}")
            first = old_last + 2**(m-2)
            print(f"new first is: {first}")
            last = first + step + (k-1)*(2**(m-1))
            print(f"new last is: {last}")

            low = sum([self.k*2**(i-1) for i in range(1, m+1)], 0)-last
            print(f"low is: {low}")
            print()

            for i in range(first, t+1, 2**(m-1)):

                # FROM i
                for j in range((i+step)+1, (i+step)+low+1):
                    edges.append((i, j, 1+abs((i+step)-j)))
                    print(f"W{1+abs((i+step)-j)}-edge drawn from {i}->{j}")

                for j in range(i+int((step+1)/2), i+step):
                    # print(j)
                    edges.append((i, j, 1+abs((i+step)-j)))
                    print(f"W{1+abs((i+step)-j)}-edge drawn from {i}->{j}")

                # FROM i+step
                for j in range(i-low, i):
                    edges.append((i+step, j, 1+abs(i-j)))
                    print(f"W{1+abs(i-j)}-edge drawn from {i+step}->{j}")

                for j in range(i+1, (i+step)-int((step-1)/2)):
                    # print(j)
                    edges.append((i+step, j, 1+abs(i-j)))
                    print(f"W{1+abs(i-j)}-edge drawn from {i+step}->{j}")

                # --------------------------------------------------------

                # FROM i
                for a in range(1, low+1):
                    # print(f"a is: {a}")
                    for j in range(i+step+1, (i+step)+low+1):
                        if (i-a, j, 1+abs((i+step)-j)+a) not in edges and (j, i-a, 1+abs((i+step)-j)+a) not in edges:
                            edges.append((i-a, j, 1+abs((i+step)-j)+a))
                            print(
                                f"W{1+abs((i+step)-j)+a}-edge drawn from {i-a}->{j}")

                    for j in range(i+int((step+1)/2), i+step):
                        if (i-a, j, 1+abs((i+step)-j)+a) not in edges and (j, i-a, 1+abs((i+step)-j)+a) not in edges:
                            edges.append((i-a, j, 1+abs((i+step)-j)+a))
                            print(
                                f"W{1+abs((i+step)-j)+a}-edge drawn from {i-a}->{j}")

                for a in range(1, int((step+1)/2)):
                    # print(f"a is: {a}")
                    for j in range(i+step+1, (i+step)+low+1):
                        if (i+a, j, 1+abs((i+step)-j)+a) not in edges and (j, i-a, 1+abs((i+step)-j)+a) not in edges:
                            edges.append((i-a, j, 1+abs((i-j)+a)))
                            print(
                                f"W{1+abs((i+step)-j)+a}-edge drawn from {i+a}->{j}")

                    for j in range(i+int((step+1)/2), i+step):
                        if (i+a, j, 1+abs((i+step)-j)+a) not in edges and (j, i-a, 1+abs((i+step)-j)+a) not in edges:
                            edges.append((i-a, j, 1+abs((i-j)+a)))
                            print(
                                f"W{1+abs((i+step)-j)+a}-edge drawn from {i+a}->{j}")

                print()
                print()
                # FROM i+step
                for a in range(1, low+1):
                    # print(f"a is: {a}")
                    for j in range(i-low, i):
                        if ((i+step)+a, j, 1+abs((i+step)-j)+a) not in edges and (j, (i+step)+a, 1+abs((i+step)-j)+a) not in edges:
                            edges.append(
                                ((i+step)+a, j, 1 + abs((i+step)-j)+a))
                            print(
                                f"W{1+abs((i+step)-j)+a}-edge drawn from {(i+step)+a}->{j}")

                    for j in range(i+1, (i+step)-int((step-1)/2)):
                        if ((i+step)+a, j, 1+abs((i+step)-j)+a) not in edges and (j, (i+step)+a, 1+abs((i+step)-j)+a) not in edges:
                            edges.append(((i+step)+a, j, 1+abs((i+step)-j)+a))
                            print(
                                f"W{1+abs((i+step)-j)+a}-edge drawn from {(i+step)+a}->{j}")

                for a in range(1, int((step+1)/2)):
                    # print(f"a is: {a}")
                    for j in range(i-low, i):
                        if ((i+step)-a, j, 1+abs((i+step)-j)+a) not in edges and (j, (i+step)-a, 1+abs((i+step)-j)+a) not in edges:
                            edges.append(((i+step)-a, j, 1+abs((i+step)-j)+a))
                            print(
                                f"W{1+abs((i+step)-j)+a}-edge drawn from {(i+step)-a}->{j}")

                    for j in range(i+1, (i+step)-int((step-1)/2)):
                        if ((i+step)-a, j, 1+abs((i+step)-j)+a) not in edges and (j, (i+step)-a, 1+abs((i+step)-j)+a) not in edges:
                            edges.append(((i+step)-a, j, 1+abs((i+step)-j)+a))
                            print(
                                f"W{1+abs((i+step)-j)+a}-edge drawn from {(i+step)-a}->{j}")

                edges.append((i, i+step, 1))
                print(f"edge drawn from {i}->{i+step}")

            return self.construct_graph(m+1, edges, first, last, step)

        self.G.add_weighted_edges_from(edges)

    def count_weight_1_edges(self, node):
        return sum(1 for _, neighbor, data in self.G.edges(node, data=True) if data["weight"] == 1)

    def sort_nodes(self):
        return sorted(self.G.nodes, key=self.count_weight_1_edges, reverse=True)

    def __str__(self):
        info_string = f"G_mi Graph Info \n"
        info_string += f"--------------------\n"
        info_string += f"n-levels: {self.n}\n"
        info_string += f"nodes per level: {self.k}\n"
        info_string += f"construction: {self.construction}\n"
        info_string += f"total nodes: {self.get_total_nodes()}\n"
        # info_string += f"weighted adjacency matrix: \n\n"
        # info_string += self.display_sub_adj(3)
        return info_string


n = 6
k = 7
TestGraph = G_mi(n, k, "General")

TestGraph.construct_graph(1, [], k+2*k+2, k+2*k, 1,)
TestGraph.populate_weights()

Gr = TestGraph.get_graph()
kr = TestGraph.get_k()

shells = [[0], list(range(1, (kr*2**0)+1))]

for i in range(1, n+1):
    start = sum(kr * (2**j) for j in range(i))
    end = start + kr * (2**i)
    shells.append(list(range(start + 1, end + 1)))

pos = {}

radii = list(range(0, TestGraph.get_n()+3))
# print(len(radii))

shell_angles = []

max_weight = max(data['weight'] for _, _, data in Gr.edges(data=True))

# Define a function to get RGBA color based on weight

# # color: darker => lower weight
# def get_rgba_color(weight, max_weight, min_opacity=0.2):
#     opacity = max(1.0 - ((weight - 1) / (max_weight - 1))
#                   * (1.0 - min_opacity), min_opacity)
#     return (0, 0, 0, opacity)


# # color: darker => higher weight
def get_rgba_color(weight, max_weight, min_opacity=0.2):
    opacity = min_opacity + \
        ((weight - 1) / (max_weight - 1)) * (1.0 - min_opacity)
    return (0, 0, 0, opacity)

# color: from the rainbow color map with varying opacity per weight
# def get_rgba_color(weight, max_weight, min_opacity=0.2):
#     normalized_weight = 1 - (weight / max_weight)
#     color = cm.rainbow(normalized_weight)
#     opacity = min_opacity + normalized_weight * (1.0 - min_opacity)
#     return (color[0], color[1], color[2], opacity)


# Apply color to each edge based on its weight
edge_colors = [get_rgba_color(data['weight'], max_weight)
               for _, _, data in Gr.edges(data=True)]

for i, shell in enumerate(shells):
    radius = radii[i]
    num_nodes = len(shell)
    angles = np.linspace(0, 2.0 * np.pi, num_nodes,
                         endpoint=False) + (np.pi/2**radius+((np.pi/5)*radius*(radius-1)))-(np.pi/2)
    shell_angles.append(angles)

    for j, node in enumerate(shell):
        pos[node] = (radius * np.cos(shell_angles[i][j]),
                     radius * np.sin(shell_angles[i][j]))

options = {
    'node_color': 'black',
    'node_size': 50,
    'width': 1,
}

# print("DEGREES")
# print(dict(Gr.degree()))

print()
print(Gr.nodes)
print()
print(Gr.edges)


adj_matrix = nx.to_numpy_array(Gr, weight='weight')
print(adj_matrix)

# savemat("adj_matrix(4).mat", {"adj_matrix": adj_matrix})

plt.figure(figsize=(7, 7))
nx.draw(Gr, pos, with_labels=False, font_size=5,
        font_weight='bold', font_color='white', edge_color=edge_colors, ** options)

# edge_color=edge_colors


# def networkx_to_tikz(graph):
#     """
#     Converts a NetworkX graph to TikZ code for LaTeX.

#     Parameters:
#         graph (networkx.Graph): The NetworkX graph to convert.

#     Returns:
#         str: The generated TikZ code.
#     """
#     tikz_code = "\\begin{tikzpicture}\n"

#     # Define node positions
#     pos = nx.spring_layout(graph)  # This layout can be changed as needed.

#     # Add nodes to TikZ code
#     for node in graph.nodes():
#         x, y = pos[node]
#         tikz_code += f"\\node ({node}) at ({x:.2f}, {y:.2f}) {{{node}}};\n"

#     # Add edges to TikZ code
#     for u, v in graph.edges():
#         tikz_code += f"\\draw ({u}) -- ({v});\n"

#     tikz_code += "\\end{tikzpicture}"
#     return tikz_code


# tikz_output = networkx_to_tikz(Gr)
# print(tikz_output)

print(TestGraph)
plt.show()
