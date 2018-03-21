# -*- coding: utf-8 -*-
"""
Created on Sun Mar 04 03:45:34 2018

@author: KevinYeung
"""

import networkx as nx
import numpy as np
import sys 
#sys.path.append('/home/kevin1024/Desktop/AGGP') #set the path 
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
    def calculate_fitness(self,gamma_basic,C_basic,L_basic):
        gamma = mc.degree_coef(self.graph)
        #beta = mc.clustering_coef(self.graph)
        C = nx.average_clustering(self.graph)
        L = nx.average_shortest_path_length(self.graph)
        print (gamma,C,L)
        fitness = fitness_func(gamma,C,L,gamma_basic,C_basic,L_basic)
        print fitness
        self.fitness = fitness
        return self.fitness

def fitness_func(gamma,C,L,gamma_basic,C_basic,L_basic):
    fitness = ((gamma-gamma_basic)**2+(C-C_basic)**2+(L-L_basic)**2)
    return fitness
#generate a population
def create_population(pop_num,node_num):
    population = []
    for i in range(pop_num):
        g = Graph(node_num)
        g.generate_random_connected_graph()    
        g.calculate_fitness(1,1,1) #3 parameter from the article
        population.append(g)
    return population



def cross_over(aGraph,bGraph,numnber_of_co):
	nodes = list(aGraph.nodes())
	sample = random.sample(nodes,numnber_of_co)
	print(sample)
	for node in sample :
		edges_a = list(nx.edges(aGraph,node))
		edges_b = list(nx.edges(bGraph,node))

		aGraph.remove_edges_from(edges_a)
		bGraph.remove_edges_from(edges_b)

		aGraph.add_edges_from(edges_b)
		bGraph.add_edges_from(edges_a)



#test: create a population, input parameter:number of graphs and number of nodes    
population = create_population(10,3000)
#sort the population by its fitness   
population.sort(cmp=None, key=lambda x:x.fitness, reverse=True)

#test
for g in population:
    print g.fitness




