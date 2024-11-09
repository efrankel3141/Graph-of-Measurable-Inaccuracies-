import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class G_mi:

    # -----------------------------------------------------
    # GRAPH INITIALIZER ->
    # [num of n-levels, nodes per level, construction-type]
    # -----------------------------------------------------
    def __init__(self, n, npl, construction):
        self.n = n
        self.construction = construction
        self.npl = npl

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
    total_nodes = 0
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
    def get_npl(self):
        return self.npl

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
    # -----------------------------------------------------

    # -----------------------------------------------------
    # GET TOTAL NUM OF NODES
    # -----------------------------------------------------
    def get_total_nodes(self):
        return self.total_nodes

    # -----------------------------------------------------
    # GENERATE GRAPH - (base)
    # adds a "central" vertex as 0, and then adds all other
    # nodes from the computed total nodes (dep. on n)
    # -----------------------------------------------------
    def initalize_graph(self):
        self.G.add_node(0)
        self.G.add_nodes_from(list(range(1, self.total_nodes)))

    # -----------------------------------------------------
    # GENERATE ADJ. MATRIX - (based off specific graph)
    # looks at the weight for each edge in the edge set for
    # and appropriately encodes it in a matrix with dim.
    # (\sum_{i=0}^n 2^{n-1})^2 or (total_nodes)^2
    # -----------------------------------------------------
    def populate_weights(self):
        self.weights = [[0 for _ in range(self.total_nodes)]
                        for _ in range(self.total_nodes)]
        for i in range(len(self.weights)):

            for j in range(len(self.weights[i])):

                if self.G.has_edge(i, j):

                    weight = self.G.get_edge_data(
                        i, j).get("weight", None)

                    self.weights[i][j] = weight

    # -----------------------------------------------------
    # GENERATE GRAPH - (base)
    # adds a "central" vertex as 0, and then adds all other
    # nodes from the computed total nodes (dep. on n)
    # -----------------------------------------------------
    def gen_total_nodes(self):
        for i in range(0, self.n):
            self.total_nodes += self.npl*(2**(i))

    def display_weights(self):
        matrix = ""
        for row in self.weights:
            matrix += str(row)+"\n"
        return matrix

    def display_sub_adj(self, r):
        sub_matrix = ""
        index = 0
        for row in self.weights:
            if index > self.npl*(2**(r)):
                sub_matrix += str(row[self.npl*(2**(r))+1:])+"\n"
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
    def construct_graph(self, k, p_nodes):
        edges = []

        if k == 1:
            for i in range(1, len(self.G.nodes)+1):
                edges.append((0, i, {"weight": 1}))
            self.G.add_edges_from(edges)

            # print("l_1:" + str(self.G.edges))
            return self.construct_graph(k+1, list(range(1, 29)))

        elif k == 2:
            for i in range(self.npl+1, self.total_nodes, 2):
                edges.append((i, i+1, {"weight": 1}))
            self.G.add_edges_from(edges)

            # print("l_2:" + str(self.G.edges))
            return self.construct_graph(k+1, list(range(1, 29)))

        elif k <= self.n:
            lower = 0
            subgraphs = []
            edges = []

            sub_A = []
            sub_B = []
            a_p = 0
            b_p = 0

            lower = sum(self.npl * (2 ** m) for m in range(k - 1, self.n))

            subgraphs = [
                list(range(j, j + 2 ** (k - 2)))
                for j in range((self.total_nodes - lower) + 1, self.total_nodes, 2 ** (k - 2))
            ]

            updated_p = [x + self.npl*(2**(k-2)) for x in p_nodes]
            print(updated_p)

            temp_p_nodes = []
            for i in range(0, len(subgraphs) - 1, 2):
                sub_A, sub_B = subgraphs[i], subgraphs[i + 1]

                a_p = next((a for a in sub_A if a in updated_p), None)
                b_p = next((b for b in sub_B if b in updated_p), None)

                if a_p is not None and b_p is not None:
                    temp_p_nodes.extend([a_p, b_p])
                    print(f"Principal nodes are: a_p: {a_p} and b_p: {b_p}")

                    # Add the edge between a_p and b_p
                    self.G.add_edge(a_p, b_p, weight=1)

                for a in sub_A:
                    if a == a_p:
                        for b in sub_B:
                            if b != b_p:
                                weight = (len(nx.shortest_path(
                                    self.G, source=b, target=b_p))-1) + 1
                                print("not b_p, yes a_p | b->b_p")
                                print(
                                    (len(nx.shortest_path(self.G, source=b, target=b_p))-1))

                                print("b: "+str(weight), str(a)+"->"+str(b))
                                edges.append((a_p, b, {"weight": weight}))
                            else:
                                continue
                    else:
                        print(a)
                        for b in sub_B:
                            if b != b_p:
                                weight = (len(nx.shortest_path(
                                    self.G, source=b, target=b_p))-1) + (len(nx.shortest_path(
                                        self.G, source=a, target=a_p))-1) + 1
                                print("not b_p, not a_p | a->b")
                                print((len(nx.shortest_path(
                                    self.G, source=b, target=b_p))-1))
                                print(" + ")
                                print((len(nx.shortest_path(
                                    self.G, source=a, target=a_p))-1))

                                print("b: "+str(weight), str(a)+"->"+str(b))
                                edges.append((a, b, {"weight": weight}))
                            else:
                                weight = (len(nx.shortest_path(
                                    self.G, source=a, target=a_p))-1) + 1

                                print("yes b_p, not a_p | a->a_p")
                                print((len(nx.shortest_path(
                                    self.G, source=a, target=a_p))-1))
                                print("a: "+str(weight), str(a)+"->"+str(b_p))
                                edges.append((a, b, {"weight": weight}))

            self.G.add_edges_from(edges)
            return self.construct_graph(k+1, temp_p_nodes)

    def count_weight_1_edges(self, node):
        return sum(1 for _, neighbor, data in self.G.edges(node, data=True) if data["weight"] == 1)

    def sort_nodes(self):
        return sorted(self.G.nodes, key=self.count_weight_1_edges, reverse=True)

    def __str__(self):
        info_string = f"G_mi Graph Info \n"
        info_string += f"--------------------\n"
        info_string += f"n-levels: {self.n}\n"
        info_string += f"nodes per level: {self.npl}\n"
        info_string += f"construction: {self.construction}\n"
        info_string += f"total nodes: {self.get_total_nodes()}\n"
        info_string += f"weighted adjacency matrix: \n\n"
        info_string += self.display_sub_adj(3)
        return info_string


options = {
    'node_color': 'black',
    'node_size': 100,
    'width': 1,
}

k = 4

TestGraph = G_mi(k, 4, "General")
TestGraph.gen_total_nodes()
TestGraph.initalize_graph()

TestGraph.construct_graph(1, list(range(1, 29)))
TestGraph.populate_weights()

Gr = TestGraph.get_graph()
npl = TestGraph.get_npl()
print("HERE")
print(Gr.get_edge_data(38, 40).get("weight", None))

# matrix_arr = np.array(TestGraph.get_weight_matrix())

# np.savetxt("matrix2.csv", matrix_arr, delimiter=",", fmt="%d")


shells = [[0], list(range(1, (npl*2**0)+1))]

for i in range(1, k+1):
    start = sum(npl * (2**j) for j in range(i))
    end = start + npl * (2**i)
    shells.append(list(range(start + 1, end + 1)))

pos = {}

radii = list(range(0, TestGraph.get_n()+2))

shell_angles = []

weight_to_color = {
    1: 'black',
    2: 'black',
    3: 'black',
}

edge_colors = [weight_to_color.get(data['weight'], 'black')
               for u, v, data in Gr.edges(data=True)]

for i, shell in enumerate(shells):
    radius = radii[i]
    num_nodes = len(shell)
    angles = np.linspace(0, 2.0 * np.pi, num_nodes,
                         endpoint=False) + np.pi/2**radius+((np.pi/5)*radius*(radius-1))
    shell_angles.append(angles)

    for j, node in enumerate(shell):
        pos[node] = (radius * np.cos(shell_angles[i][j]),
                     radius * np.sin(shell_angles[i][j]))

nx.draw(Gr, pos, edge_color=edge_colors, with_labels=False, **options)

print(TestGraph)
plt.show()
