from node import Node

class Connection:
    def __init__(self):
        self.id = None
        self.src = None
        self.dst = None

        self.weight = 1
        self.enabled = True

    def fforward(self, input):
        return input * self.weight
