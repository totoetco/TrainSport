# -*- coding: utf-8 -*-
"""
Created on Sun Mar 04 03:45:34 2018

@author: KevinYeung
"""

import networkx as nx
import numpy as np
import sys 
sys.path.append('/home/kevin1024/Desktop/AGGP') 
import Mcoef as mc

class Graph:
    def __init__(self,node_num):
        self.node_num = node_num #number of nodes
        #self.edge_num = edge_num
        self.graph = nx.Graph() #a empty graph
        self.fitness = -1 #set fitness, defaut is -1
        #self.degree = -1 #init degree
    def show_graph(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True)
        
    
    def generate_random_connected_graph(self):
        nodes = range(self.node_num)#create a vector of nodes,length=nodes number
        self.graph.add_nodes_from(nodes)#create a graph with all nodes, but no edges
    	#k = 0
    	while (nx.is_connected(self.graph) != True):
             new_edges = np.random.choice(nodes, size=(10,2)) #randomly generate 10 edges
             self.graph.add_edges_from(new_edges)
             #k+=1
        self.degree = self.graph.degree()
        return self.graph
    def calculate_fitness(self):
        gamma_basic = -1 #set standard gamma
        C_basic = -1
        L_basic = -1
        gamma = mc.degree_coef(self.graph)
        #beta = mc.clustering_coef(self.graph)
        C = nx.average_clustering(self.graph)
        L = nx.average_shortest_path_length(self.graph)
        fitness = (1/3)*(abs(gamma-gamma_basic)+abs(C-C_basic)+abs(L-L_basic))
        self.fitness = fitness
        return self.fitness

#generate a population
population = []
for i in range(2,5):
    g = Graph(i)
    g.generate_random_connected_graph()    
    g.calculate_fitness()
    population.append(g)

#sort the population by its fitness   
population.sort(cmp=None, key=lambda x:x.fitness, reverse=True)

#test
for g in population:
    print g.fitness




