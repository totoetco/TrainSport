# -*- coding: utf-8 -*-
"""
Created on Sun Mar 04 03:45:34 2018

@author: KevinYeung
"""

import networkx as nx
import numpy as np
import sys 
#sys.path.append('/home/kevin1024/Desktop/AGGP') #set the path 
import parameters as mc
import random
import matplotlib.pyplot as plt


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
        plt.show()
        
    def __repr__(self):
        return("Graphe de "+str(len(list(self.graph.edges())))+" edges, fitness de "+str(self.fitness))

    def generate_random_connected_graph(self):
        nodes = range(self.node_num)#create a vector of nodes,length=nodes number
        self.graph.add_nodes_from(nodes)#create a graph with all nodes, but no edges
          #k = 0
        while (nx.is_connected(self.graph) != True):
            new_edges = np.random.choice(nodes, size=(10,2)) #randomly generate 10 edges
            self.graph.add_edges_from(new_edges)
             #k+=1
        #self.degree = self.graph.degree()
        return self.graph

    def generate_barabasi_graph(self):
        self.graph=nx.barabasi_albert_graph(self.node_num, 5)

    def calculate_fitness(self,gamma_basic,C_basic,L_basic):
        gamma = mc.degree_coef(self.graph)
        #beta = mc.clustering_coef(self.graph)
        C = nx.average_clustering(self.graph)
        L = nx.average_shortest_path_length(self.graph)
        fitness = fitness_func(gamma,C,L,gamma_basic,C_basic,L_basic)
        self.fitness = fitness
        return self.fitness

    def mutation_of_a_graph(self,number_to_change,method = None):
        if method == None :
            a = int(round(random.random()*number_to_change))
            b = number_to_change-a
            a_changed = 0
            b_changed = 0
            while a_changed < a:
                r_edge = random.sample(self.graph.edges(),1)
                self.graph.remove_edge(*r_edge[0])
                if(nx.is_connected(self.graph)):
                    a_changed+=1
                else:
                    self.graph.add_edge(*r_edge[0])
            while b_changed < b:
                random_node1 = random.choice(np.array(nx.nodes(self.graph)))
                random_node2 = random.choice(np.array(nx.nodes(self.graph)))
                if (random_node2 not in self.graph.neighbors(random_node1)):
                    self.graph.add_edge(random_node1,random_node2)
                    b_changed+=1

def fitness_func(gamma,C,L,gamma_basic,C_basic,L_basic):
    fitness = -((gamma-gamma_basic)**2+(C-C_basic)**2+(L-L_basic)**2)
    return fitness
#generate a population

def create_population(pop_num,node_num, method="random"):
    population = []
    for i in range(pop_num):
        g = Graph(node_num)

        if method=="random":
            g.generate_random_connected_graph()   
        elif method=="barabasi":
            g.generate_barabasi_graph()
        else: 
            raise NameError('Wrong method.')

        g.calculate_fitness(1,1,1) #3 parameter from the article
        population.append(g)

    return population




def cross_over(aGraph,bGraph,numnber_of_co):
    nodes = list(aGraph.nodes())
    n = 0
    while n < numnber_of_co:

        node = random.choice(nodes)
        edges_a = list(nx.edges(aGraph,node))
        edges_b = list(nx.edges(bGraph,node))


        aGraph.remove_edges_from(edges_a)
        bGraph.remove_edges_from(edges_b)

        aGraph.add_edges_from(edges_b)
        bGraph.add_edges_from(edges_a)

        if nx.is_connected(aGraph) and nx.is_connected(bGraph):
            n+=1

        else:
            aGraph.add_edges_from(edges_a)
            bGraph.add_edges_from(edges_b)

            aGraph.remove_edges_from(edges_b)
            bGraph.remove_edges_from(edges_a)

    return aGraph, bGraph

    


#test: create a population, input parameter:number of graphs and number of nodes    

#population = create_population(2,10)
#mut = population[0].mutation_of_a_graph(3)

#sort the population by its fitness  
#population.sort(cmp=None, key=lambda x:x.fitness, reverse=True)
#print(population)
'''
population[0].show_graph()
population[1].show_graph()
population[0].graph , population[0].graph = cross_over(population[0].graph, population[1].graph, 2)
print(a.edges())
print(b.edges())
ag = Graph(5)
ag.graph = a
bg = Graph(5)
bg.graph = b
ag.show_graph()
bg.show_graph()
'''

#test
#for g in population:
    #print g.fitness


def main(nb_nodes, nb_graph, nb_select, p_mute, p_co, nb_mutation, nb_co) :
    population = create_population(nb_graph, nb_nodes)
    print(population, '\n')

    i = 0
    while i < 20 :

        population.sort(key=lambda x:x.fitness, reverse=True)

        ## Selection!!
        good = population[:nb_select]
        bad = population[nb_select:]
        

        ## crossing over!!
        for G in good:

            if random.random()>p_co:
                s=int(random.choice(np.linspace(0, len(good))))
                G.graph, population[s].graph = cross_over(G.graph, population[s].graph, nb_co)

     

        ## mutation!!
        for G in population: 

            if random.random()>p_mute:
                G.mutation_of_a_graph(nb_mutation)
            G.calculate_fitness(1,1,1)
     


        i+=1
    return(population[0])

best = main(20, 30, 10, 0.5, 0.2, 2, 4)
print("\nLe meilleur graphe est: ", best, "Il a pour coefficients C, gamma et L: ", nx.average_clustering(best.graph),  mc.degree_coef(best.graph), nx.average_shortest_path_length(best.graph),'\n')



