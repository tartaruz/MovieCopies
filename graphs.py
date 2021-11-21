from os import read
import numpy
import pandas
from classes import *
import random
from collections import Counter

def readFile():
    f = open("./res.txt", "r")
    structures = [line.replace("\n","").split("-") for line in f if len(line.split("-"))==3]
    return structures

def makeNodes():
    file = readFile()
    struct = [Movie(el[0]) for el in file]
    position = { el.name:i for i,el in enumerate(struct)}
    for i,node in enumerate(struct):
        try:
            sim_postition = position[file[i][1]]
            target_node = struct[sim_postition]
            
        except KeyError as e:
            target_node = None

        node.similar = (target_node)
    for mov in struct:
        
        if mov.similar == None:
            struct.remove(mov)
        
        if mov == None:
            struct.remove(mov)
        
    return struct

def pageRank(struct):
    jumps = 0
    damping = 0.85
    selected = random.choice(struct)
    while jumps < 100000000:
        jumps +=1
        r = random.random()
        if r<=damping and not selected == None:
            next = selected.similar    
        else:
            next = random.choice(struct)
        
        if not next == None:
            next.point()
        selected = next
    for n in struct:
        if n.importance>0:
            n.importance = n.importance*100/jumps
    
    struct.sort(key=lambda x:x.importance, reverse=True)
    
    for mov in struct[:100]:
        print(f"P:{mov.importance}\t{mov.name}")

struct = makeNodes()
c = Counter([node.similar.name for node in struct if not node.similar == None])

print(c.most_common(100))
pageRank(struct)
# print(c())