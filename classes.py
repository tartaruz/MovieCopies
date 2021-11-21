class Movie():
    def __init__(self, name):
        self.name = name
        self.similar = None
        self.importance = 0.0
        
    def addSimilar(self, movie):
        self.similar = movie

    def point(self):
        self.importance += 1


# class Graph:
#     def __init__(self):
#         self.nodes = []

#     def addNode(self, target, similar, percentage):
#         node = Movie(target)
#         node.addSimilar(similar, percentage)
#         self.nodes.append(node)


#     def findNode(self):
#         pass