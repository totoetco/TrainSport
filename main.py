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
import time
import math

class Graph:

    def __init__(self,node_num):
        self.node_num = node_num #number of nodes
        #self.edge_num = edge_num
        self.graph = nx.Graph() #a empty graph
        self.fitness = -1 #set fitness, defaut is -1
        self.C=None
        self.gamma=None
        self.L=None
        #self.degree = -1 #init degree

    def show_graph(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True)
        plt.show()
        
    def __repr__(self):
        return("Graphe de "+str(len(list(self.graph.edges())))+" edges, fitness: "+str(self.fitness)+ " C,L,gamma: "+str(self.C)+", "+str(self.L)+", "+str(self.gamma))

    def generate_random_connected_graph(self):
        nodes = range(self.node_num)#create a vector of nodes,length=nodes number
        self.graph.add_nodes_from(nodes)#create a graph with all nodes, but no edges
          #k = 0
        while (nx.is_connected(self.graph) != True):
            new_edges = np.random.choice(nodes, size=(10,2)) #randomly generate 10 edges
            self.graph.add_edges_from(new_edges)
             #k+=1
        #self.degree = self.graph.degree()
        #return self.graph

    def generate_barabasi_graph(self):
        self.graph=nx.barabasi_albert_graph(self.node_num, 1)

    def calculate_fitness(self,gamma_basic,C_basic,L_basic):
        gamma = mc.degree_coef(self.graph)
             #beta = mc.clustering_coef(self.graph)
        C = nx.average_clustering(self.graph)

        L = nx.average_shortest_path_length(self.graph)


        fitness = fitness_func(gamma,C,L,gamma_basic,C_basic,L_basic)
        self.C = C
        self.gamma = gamma
        self.L = L
        self.fitness = fitness
        #return self.fitness

    def mutation_v2(self, number_to_change,method = None):
        if(method == None):
            #a = int(round(random.random()*number_to_change)) #remove
            #b = number_to_change-a #add
            a = number_to_change
            b = 0
        while True:
            graph_change = self.graph.copy()
            remove_edge = random.sample(graph_change.edges(),a)
            add_edge = np.random.choice(self.node_num,size=(b,2),replace=False)
            #print ('remove',remove_edge)
            #print ('add',add_edge)
            graph_change.remove_edges_from(remove_edge)            
            graph_change.add_edges_from(add_edge)
            if nx.is_connected(graph_change):
                break
        self.graph = graph_change   

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
                    ## Proposition to solve the problem of impossible mutations
                    a-=1
                    b+=1
                    ##

            while b_changed < b:
                random_node1 = random.choice(np.array(nx.nodes(self.graph)))
                random_node2 = random.choice(np.array(nx.nodes(self.graph)))
                if (random_node2 not in self.graph.neighbors(random_node1)):
                    self.graph.add_edge(random_node1,random_node2)
                    b_changed+=1


        if (method == "remove"):
            for i in range(number_to_change):
                random_node = random.choice(np.array(nx.nodes(self.graph)))
            if(self.graph.degree(random_node)>2):
                remove_edge = random.sample(self.graph.edges(random_node),1)
                self.graph.remove_edges_from(remove_edge)
            else : 
                random_node2 = random.choice(np.array(nx.nodes(self.graph)))
                if (random_node2 not in self.graph.neighbors(random_node1)):
                    self.graph.add_edge(random_node1,random_node2)  


def fitness_func(gamma,C,L,gamma_basic,C_basic,L_basic):
    fitness = math.sqrt(((1.0/3)*(gamma-gamma_basic)**2+(1.0/3)*(C-C_basic)**2+(1.0/3)*(L-L_basic)**2))
    return fitness
#generate a population

def create_population(pop_num,node_num, method):
    population = []
    for i in range(pop_num):
        g = Graph(node_num)

        if method=="random":
            g.generate_random_connected_graph()   
        elif method=="barabasi":
            g.generate_barabasi_graph()
        else: 
            raise NameError('Wrong method.')
        g.calculate_fitness(1,1,1) #3 parameter from the article 1.83,0.835,3.5
        population.append(g)
    return population


def cross_over(aGraph,bGraph,numnber_of_co):

    nodes = list(aGraph.nodes())
    num = numnber_of_co
    n = 0

    aG=aGraph.copy()
    
    ## Proposition to avoid not compatible graphs: if we try N times 
    # to cross over nodes and it doesn't work, we decrease the number of 
    # co to apply. 

    fail = 0
    while n < num:

        node = random.choice(nodes)
        edges_a = list(nx.edges(aGraph,node))*1
        edges_b = list(nx.edges(bGraph,node))*1


        aG.remove_edges_from(edges_a)
        aG.add_edges_from(edges_b)



        if nx.is_connected(aG):
            n+=1

        else:
            fail+=1
            aG=aGraph.copy()
        ##
        if fail==10:
            num-=1
            fail = 0
        ##



    return aG#, bGraph




def main(nb_nodes, nb_graph, nb_select, p_mute, p_co, nb_mutation, nb_co) :

    ## Graph population generation
    t1=time.time()
    population = create_population(nb_graph, nb_nodes, "barabasi")
    print("temps generation", time.time()-t1)


    i = 0
    fitnesses = []
    coef=[[], [], []]

    # Loop begins
    while i < 20 :

        # Population is sorted
        t1=time.time()
        population.sort(key=lambda x:x.fitness, reverse=False)
        print("fitness: ", population[0].fitness)
        fitnesses.append(population[0].fitness)
        coef[0].append(population[0].C)
        coef[1].append(population[0].L)
        coef[2].append(population[0].gamma)
        #print(population, '\n')
        print(" temps sort", time.time()-t1)

        # Old test I used to check if graphs are connected
        #for G in population:
        #    if (nx.is_connected(G.graph)):
        #        pass
        #    else:
        #        print("graph deconnecte, sort")

     
        #### crossing over!!
        t1=time.time()
        for G in population[nb_select:]:
            if random.random()<p_co:
                s=int(random.choice(np.linspace(0, nb_select)))
                G.graph = cross_over(G.graph, population[s].graph, nb_co) #, population[s].graph
        print(" temps co", time.time()-t1)


        
        #### mutation!!
        t1=time.time()
        for G in population: 
            if random.random()<p_mute:
                G.mutation_of_a_graph(nb_mutation)
        print(" temps mutation", time.time()-t1)


        #### Fitness computation!!
        t1=time.time()
        for G in population: 
            G.calculate_fitness(1,1,1) #3 parameter from the article 1.83,0.835,3.5
        print(" temps fitness", time.time()-t1)
        print("coefficients (C, L, gamma) :", population[0].C,population[0].L,population[0].gamma)
        i+=1



    return(population[0], fitnesses, coef)


def plot_evol(fitnesses, coef):

    plt.plot(np.arange(1, len(fitnesses)+1), fitnesses,label = 'fitness')
    plt.plot(np.arange(1, len(fitnesses)+1), coef[0],label = 'C')
    plt.plot(np.arange(1, len(fitnesses)+1), coef[1],label = 'L')
    plt.plot(np.arange(1, len(fitnesses)+1), coef[2],label = 'gamma')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=4, borderaxespad=-6)
    plt.show()

## Test ##
'''
best, f, c = main(100, 30, 10, 1, 1, 20, 4)
print("\nLe meilleur graphe est: ", best, "Il a pour coefficients C, gamma et L: ", nx.average_clustering(best.graph),  mc.degree_coef(best.graph), nx.average_shortest_path_length(best.graph),'\n')
plot_evol(f, c)
'''
## Paramatric exploration: 
# Function that computes the mean of the difference from the initial fitness 
# during execution
def moy_diff(f):
    
    m = f[0]
    diff=abs(np.asarray(f)-m)

    return(np.mean(diff), math.sqrt(np.var(diff)))

# Launch 5 execution of code with given parameter, and returns statistical quality results

def launch_exploration(nb_select, p_mute, p_co, nb_mutation, nb_co):
    mean = []
    var = []
    for i in range(1):
        best, f, c = main(100, 30, nb_select, p_mute, p_co, nb_mutation, nb_co)
        m, v = moy_diff(f)
        mean.append(m)
        var.append(v)

    plot_evol(f, c)
    mean = np.asarray(mean)
    var = np.asarray(var)
    
    return(np.mean(mean), np.mean(var), np.var(mean))

m, mv, v = launch_exploration(10, 1, 1, 20, 4)

print (u"Pour les valeurs de paramètres données, l'écart moyen à la valeur initiale est de: "+ str(m) +"la variance moyenne de cet écart est de "+ str(mv) + "et la variance inter-populations de "+ str(v)) 
