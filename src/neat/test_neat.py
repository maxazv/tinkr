from genome.genome import Genome


papa = Genome(2, 2)
for i in range(4):
    papa.add_node('h')

papa.add_connection(papa.nodes[1], papa.nodes[3], True)
papa.add_connection(papa.nodes[2], papa.nodes[4], True)
papa.add_connection(papa.nodes[2], papa.nodes[4], True)
papa.add_connection(papa.nodes[6], papa.nodes[3], True)

mama = Genome(2, 2)
for i in range(2):
    mama.add_node('h')

mama.add_connection(mama.nodes[1], mama.nodes[4], True)
mama.add_connection(mama.nodes[2], mama.nodes[4], True)
mama.add_connection(mama.nodes[1], mama.nodes[3], True)
mama.add_connection(mama.nodes[1], mama.nodes[5], True)

child = Genome.crossover(papa, mama)
print(len(child.connections))
