# -*- coding: utf-8 -*-
"""
Created on Sun Mar 04 03:45:34 2018
"""

import networkx as nx
import numpy as np
import sys 
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

	def calculate_fitness(self,gamma_basic,C_basic,L_basic, wG=1.0/3, wC=1.0/3, wL=1.0/3):
		gamma = mc.degree_coef(self.graph)
			 #beta = mc.clustering_coef(self.graph)
		C = nx.average_clustering(self.graph)

		L = nx.average_shortest_path_length(self.graph)


		fitness = fitness_func(gamma,C,L,gamma_basic,C_basic,L_basic, wG, wC, wL)
		self.C = C
		self.gamma = gamma
		self.L = L
		self.fitness = fitness
		#return self.fitness   

	def mutation_of_a_graph(self,number_to_change,method = None):
		if method == None :
			a = int(round(random.random()*number_to_change))
			b = int(round(random.random()*number_to_change))

			#a = int(round(random.random()*200))
			#b = int(round(random.random()*200))
			
			'''
			a = 0
			b = number_to_change
			''' 

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
					##

			while b_changed < b:
				random_node1 = random.choice(np.array(nx.nodes(self.graph)))
				random_node2 = random.choice(np.array(nx.nodes(self.graph)))
				if (random_node2 not in self.graph.neighbors(random_node1)):
					self.graph.add_edge(random_node1,random_node2)
					b_changed+=1

def fitness_func(gamma,C,L,gamma_basic,C_basic,L_basic, wG, wC, wL):
	#fitness = math.sqrt((wG*(gamma-gamma_basic)**2+wC*(C-C_basic)**2+wL*(L-L_basic)**2))
	fitness = (wG*abs((gamma-gamma_basic)/gamma_basic)+wC*abs((C-C_basic)/C_basic)+wL*abs((L-L_basic)/L_basic))
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
		#g.calculate_fitness(1.83,0.835,3.5) #3 parameter from the article 1.83,0.835,3.5
		g.calculate_fitness(1.07, 0.835, 2.18)
		population.append(g)
	return population


def cross_over(aGraph,bGraph,numnber_of_co):

	nodes = list(aGraph.nodes())
	num = np.random.choice(np.arange(0,numnber_of_co+1))
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




def main(nb_nodes, nb_graph, nb_select, p_mute, p_co, nb_co) :

	## Graph population generation
	t1=time.time()
	population = create_population(nb_graph, nb_nodes, "barabasi")
	print("temps generation", time.time()-t1)


	i = 0
	fitnesses = []
	coef=[[], [], []]

	## Weights
	'''
	wG = 1.0/3
	wC = 1.0/3
	wL = 1.0/3
	'''
	Gamma_target = 1.07 #1.83
	C_target = 0.835 #0.835
	L_target = 2.18 #3.5

	# Loop begins
	number_of_generations = 50
	while i < number_of_generations :

		# Population is sorted
		t1=time.time()
		population.sort(key=lambda x:x.fitness, reverse=False)
		#print("fitness: ", population[0].fitness)
		fitnesses.append(population[0].fitness)
		coef[0].append(population[0].C)
		coef[1].append(population[0].L)
		coef[2].append(population[0].gamma)
		#print(population, '\n')
		#print(" temps sort", time.time()-t1)

		# Old test I used to check if graphs are connected
		#for G in population:
		#	 if (nx.is_connected(G.graph)):
		#		 pass
		#	 else:
		#		 print("graph deconnecte, sort")
		
		#### pushing the best graph to the end !
		if(i%3==0):
			cp = population[0].graph.copy()
			population[29].graph = cp

		#### mutation!! New idea: The current first graph remains untouch - the fitness will always decrease like that!
		t1=time.time()
		nb_mutation = int(round((population[0].node_num/4)*0.9683237**i))
		for G in population[1:]: 
			if random.random()<p_mute:
				G.mutation_of_a_graph(nb_mutation)
		#print(" temps mutation", time.time()-t1)
		
		#### crossing over!!
		t1=time.time()
		for G in population[nb_select:]:
			if random.random()<p_co:
				s=int(random.choice(np.linspace(0, nb_select)))
				G.graph = cross_over(G.graph, population[s].graph, nb_co) #, population[s].graph
		#print(" temps co", time.time()-t1)
		
		#### Fitness computation!!
		t1=time.time()
		for G in population: 
			G.calculate_fitness(Gamma_target, C_target, L_target) # wG, wC, wL) #3 parameter from the article 1.83,0.835,3.5

		
		# Conversion to 100 nodes graphs: (gamma: 1.07, C: 0.835, L: 2.18)
		#print(" temps fitness", time.time()-t1)
		#print('coefficients (C, L, gamma) :', population[0].C,population[0].L,population[0].gamma)
		step = i*100/50
		#print('Avancement de :',step, '%')
		time2 = time.time() - time1
		#print('temps exection :', round(time2),'secondes')
		# Update weigths
		'''
		Tot=abs(population[0].C-C_target)+ abs(population[0].L-L_target)+abs(population[0].gamma-Gamma_target)
		wG=abs(population[0].gamma-Gamma_target)/Tot
		wC=abs(population[0].C-C_target)/Tot
		wL=abs(population[0].L-L_target)/Tot
		'''

		i+=1



	return(population[0], fitnesses, coef)


def plot_evol(fitnesses, coef):

	plt.semilogy(np.arange(1, len(fitnesses)+1), fitnesses,label = 'fitness')
	plt.semilogy(np.arange(1, len(fitnesses)+1), coef[0],label = 'C')
	plt.semilogy(np.arange(1, len(fitnesses)+1), coef[1],label = 'L')
	plt.semilogy(np.arange(1, len(fitnesses)+1), coef[2],label = 'gamma')
	plt.legend(bbox_to_anchor=(1.05, 1), loc=4, borderaxespad=-6)
	plt.show()

## Test ##
'''
best, f, c = main(100, 30, 10, 1, 1, 20, 4)
print("\nLe meilleur graphe est: ", best, "Il a pour coefficients C, gamma et L: ", nx.average_clustering(best.graph),	mc.degree_coef(best.graph), nx.average_shortest_path_length(best.graph),'\n')
plot_evol(f, c)
'''
## Paramatric exploration: 
# Function that computes the mean of the difference from the initial fitness 
# during execution
def moy_diff(f):
	
	m = f[0]
	diff=float(m)/np.asarray(f)

	return(np.mean(diff), math.sqrt(np.var(diff)))

# Launch 5 execution of code with given parameters, and returns statistical quality results

def launch_exploration(nb_select, p_mute, p_co, nb_co):
	global time1
	time1 = time.time()
	mean = []
	var = []
	for i in range(4):
		best, f, c = main(100, 30, nb_select, p_mute, p_co, nb_co)


		#best.show_graph()
		#nx.write_adjlist(best,"best.adjlist")
		m, v = moy_diff(f)
		mean.append(m)
		var.append(v)

	#plot_evol(f, c)
	mean = np.asarray(mean)
	var = np.asarray(var)
	
	return(np.mean(mean), np.mean(var), np.var(mean))


tab = []

for p_co in [0.1, 0.3, 0.5, 0.7, 0.9]:
    for p_mute in [0.1, 0.3, 0.5, 0.7,0.9]:
        #print("Avancée: ", p_co, p_mute)

        m, mv, v = launch_exploration(15, p_mute, p_co, 15)
        tab.append(m)

print(tab)

tab_select=[]


for sel in [3, 6, 10, 15, 20]:
    m, mv, v = launch_exploration(sel, 0.5, 0.5, 15)
    tab.append(m)
print(tab_select)

#m, mv, v = launch_exploration(15, 0.5, 0.5, 15)
#print (u"Pour les valeurs de paramètres données, l'écart moyen à la valeur initiale est de: "+ str(m) +"la variance moyenne de cet écart est de "+ str(mv) + "et la variance inter-populations de "+ str(v)) 
