import sys
from .node import Node

class Connection:
    def __init__(self, src, dst, innov, enabled):
        self.src = src
        self.dst = dst

        self.innov = innov

        self.weight = 1
        self.enabled = enabled

    def contains(self, node1, node2):
        if node1 == self.src and node2 == self.dst or node2 == self.src and node1 == self.dst:
            return True
        return False

