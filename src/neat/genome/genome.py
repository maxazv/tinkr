from .node import Node
from .connections import Connection
from .innovation_gen import Innovation

import random
import numpy as np


class Genome:
    def __init__(self, input_n, output_n):
        self.inp_size, self.out_size = input_n, output_n
        self.nodes = {}
        self.connections = {}

        self.species = None
        self.innov_conns = Innovation()
        self.innov_nodes = Innovation()

        self.init()

    def init(self):
        for i in range(self.inp_size):
            n = Node(self.innov_nodes.get_innov_num(), 'i')
            self.nodes[n.innov] = n
        for i in range(self.out_size):
            n = Node(self.innov_nodes.get_innov_num(), 'o')
            self.nodes[n.innov] = n

    def add_connection(self, n1, n2, enabled):
        id = self.innov_conns.get_innov_num()
        self.connections[id] = Connection(n1, n2, id, enabled)

    def add_node(self, node_type):
        id = self.innov_nodes.get_innov_num()
        self.nodes[id] = Node(id, node_type)

    def add_connection_mutation(self):
        node1 = self.nodes[random.randint(1, len(self.nodes))]

        if node1.type == 'i':
            node2 = self.nodes[random.randint(self.inp_size, len(self.nodes))]
        elif node1.type == 'o':
            node2 = self.nodes[random.randint(1, len(self.nodes) - self.out_size)]
        else:
            node2 = self.nodes[random.randint(1, len(self.nodes))]
        if node2.innov == node1.innov:
            pass    # TODO: Supposed to try again

        rev = False
        if node1.type == 'o' and node2.type == 'h':
            rev = True
        elif node1.type == 'o' and node2.type == 'i':
            rev = True
        elif node1.type == 'h' and node2.type == 'i':
            rev = True

        for conn in self.connections:
            if self.connections[conn].contains(node1, node2):
                return

        new_conn = Connection(node2 if rev else node1, node1 if rev else node2, self.innov_conns.get_innov_num(), True)
        new_conn.weight = np.random.normal(.5, .25)
        self.connections[new_conn.innov] = new_conn

    def add_node_mutation(self):
        loc = random.randint(1, len(self.connections))
        self.connections[loc].enabled = False

        new_node = Node(len(self.nodes), 'h')
        conn_to_new = Connection(self.connections[loc].src, new_node, self.innov_conns.get_innov_num(), True)
        conn_from_new = Connection(new_node, self.connections[loc].dst, self.innov_conns.get_innov_num(), True)

        conn_to_new.weight = 1
        conn_from_new.weight = self.connections[loc].weight

        self.nodes[self.innov_nodes.get_innov_num()] = new_node
        self.connections[conn_to_new.innov] = conn_to_new
        self.connections[conn_from_new.innov] = conn_from_new

    def mutate_weight(self):
        for conn in self.connections:
            rand = random.randint(0, 10)
            if rand <= 9:
                self.connections[conn].weight *= np.random.uniform(-2, 2)     # FIXME: Maybe other random
            else:
                self.connections[conn].weight = np.random.uniform(-2, 2)

    @staticmethod
    def avg_weight_diff(g1, g2):    # TODO: TEST
        count = 0
        weights_diff = 0
        for conn in g1.connections:
            if conn in g2.connections:
                weights_diff += abs(g1.connections[conn].w - g2.connections[conn])
                count += 1
        return count, weights_diff/count

    @staticmethod
    def count_disj_exc_genes(g1, g2):   # TODO: TEST
        count_disj = count_exc = 0
        max_key1 = max(g1, key=g1.connections.get)
        max_key2 = max(g2, key=g2.connections.get)

        for conn in max_key1 if max_key1 > max_key2 else max_key2:
            if conn not in g2.connections and conn < max_key2 and conn not in g1.connections and conn < max_key1:
                count_disj += 1
            elif conn not in g2.connections and conn > max_key2 and conn not in g1.connections and conn > max_key1:
                count_exc += 1

        return count_disj, count_exc

    @staticmethod
    def comp_distance(g1, g2, c1, c2, c3):
        disj, exc = Genome.count_disj_exc_genes(g1, g2)
        match, weight_diff = Genome.avg_weight_diff(g1, g2)
        return exc*c1 + disj*c2 + weight_diff*c3

    @staticmethod
    def crossover(par1, par2):      # TODO: Test properly
        child = Genome(0, 0)

        child.nodes = par1.nodes

        for conn in par1.connections:
            if conn in par2.connections:
                rand = random.randint(0, 1)
                child.connections[conn] = par1.connections[conn] if rand else par2.connections[conn]
            else:
                child.connections[conn] = par1.connections[conn]
        return child

    def conv_to_nn(self):
        pass
