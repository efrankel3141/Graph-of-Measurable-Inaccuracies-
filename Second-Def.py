import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


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
    def construct_graph(self, m, edges):
        t = self.get_total_nodes() - 1
        self.G.add_node(0)
        self.G.add_nodes_from(list(range(1, t+1)))
        print(self.G.nodes)
        print(edges)

        if m == 1:
            print("REACHED-1")
            for i in range(1, t+1):
                edges.append((0, i))
            return self.construct_graph(m+1, edges)

        elif m == 2:
            print("REACHED-2")
            for i in range(k+2*(0)+1, t, 2):
                edges.append((i, i+1))
            return self.construct_graph(m+1, edges)

        elif m == 3:
            j = self.k*(2**(m-1)-1)+2
            print(f"REACHED-{m}")
            print(f"j is {j}")
            for i in range(j, t, 4):
                print(f"i is: {i}")
                edges.append((i, i+(2**(m-2)-1)))
            return self.construct_graph(m+1, edges)

        elif m <= self.n:
            j = self.k*(2**(m-1)-1)+(2**(m-2)-1)
            print(f"REACHED-{m}")
            print(f"j is {j}")
            step = 2**(m-3)+2**(m-2)
            print(step)
            for i in range(j, t, step):
                print(f"i is: {i}")
                edges.append((i, i+(2**(m-2)-1)))
            return self.construct_graph(m+1, edges)

        self.G.add_edges_from(edges)

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
        info_string += f"weighted adjacency matrix: \n\n"
        info_string += self.display_sub_adj(3)
        return info_string


options = {
    'node_color': 'black',
    'node_size': 80,
    'width': 1,
}

n = 5
k = 4

TestGraph = G_mi(n, k, "General")

TestGraph.construct_graph(1, [])
TestGraph.populate_weights()

Gr = TestGraph.get_graph()
kr = TestGraph.get_k()
print("HERE")

shells = [[0], list(range(1, (kr*2**0)+1))]

for i in range(1, k+1):
    start = sum(kr * (2**j) for j in range(i))
    end = start + kr * (2**i)
    shells.append(list(range(start + 1, end + 1)))

pos = {}

radii = list(range(0, TestGraph.get_n()+3))
print("CHECK")
print(len(radii))

shell_angles = []

# weight_to_color = {
#     1: 'black',
#     2: 'black',
#     3: 'black',
# }

# edge_colors = [weight_to_color.get(data['weight'], 'black')
#                for u, v, data in Gr.edges(data=True)]

for i, shell in enumerate(shells):
    radius = radii[i]
    num_nodes = len(shell)
    angles = np.linspace(0, 2.0 * np.pi, num_nodes,
                         endpoint=False) + (np.pi/2**radius+((np.pi/5)*radius*(radius-1)))-(np.pi/2)
    shell_angles.append(angles)

    for j, node in enumerate(shell):
        pos[node] = (radius * np.cos(shell_angles[i][j]),
                     radius * np.sin(shell_angles[i][j]))

plt.figure(figsize=(7, 7))
nx.draw(Gr, pos, with_labels=True, font_size=5,
        font_weight='bold', font_color='white', ** options)

# edge_color=edge_colors

print(TestGraph)
plt.show()
