#! /usr/bin/env python
# -*- coding:utf-8 -*-

import networkx as nx
from networkx.utils import not_implemented_for
import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
import collections
import time 

#generating a null adjacency matrix

def generate_random_connected_graph(nbnodes):
	mynet = np.zeros((nbnodes,nbnodes))
	len1 = int(np.sqrt(np.size(mynet)))
	x = range(0,len1,1)
	y = range(0,len1,1)
	G = nx.from_numpy_matrix(np.array(mynet)) 
	k = 0
	#filling the adjacency matrix. While it is not connected, we continue to add an edge
	while (nx.is_connected(G) != True):
		i = np.random.choice(x)
		j = np.random.choice(x)
		if(mynet[i,j] != 1):
			mynet[i,j] = 1
			G = nx.from_numpy_matrix(np.array(mynet)) 
			k+=1
	return(G)


#draws the histogram of degree distribution and the graph
def drawhistandgraph(aGraph):
	degree_sequence = sorted([d for n, d in aGraph.degree()], reverse=True)  # degree sequence
	# print "Degree sequence", degree_sequence
	degreeCount = collections.Counter(degree_sequence)
	deg, cnt = zip(*degreeCount.items())

	fig, ax = plt.subplots()
	plt.bar(deg, cnt, width=0.80, color='b')

	plt.title("Degree Histogram")
	plt.ylabel("Count")
	plt.xlabel("Degree")
	ax.set_xticks([d + 0.4 for d in deg])
	ax.set_xticklabels(deg)

	# draw graph in inset
	plt.axes([0.4, 0.4, 0.5, 0.5])
	Gcc = sorted(nx.connected_component_subgraphs(aGraph), key=len, reverse=True)[0]
	pos = nx.spring_layout(aGraph)
	plt.axis('off')
	nx.draw_networkx_nodes(aGraph, pos, node_size=20)
	nx.draw_networkx_edges(aGraph, pos, alpha=0.4)
	plt.show()

def studytimeexec(nbnodes,repetitionforeachnumberofnodes):
	times = []
	nb_steps = []
	a = 800
	avancee = 0.0
	print(avancee)
	for i in range(1):
		avancee += 1
		init = time.time()
		nb_step = generate_random_connected_graph(a)
		nb_steps.append(nb_step)
		end = time.time()
		times.append(end-init)
		print((avancee/10.0)*100)
	print(a,np.mean(times),np.std(times),np.mean(nb_steps),np.std(nb_steps))

def remove_edges_to_graph_without_disconnect(aGraph, numberofedges):
	k=0
	while k < numberofedges:
		random_node1 = random.choice(np.array(nx.nodes(aGraph)))
		random_node2 = random.choice(np.array(nx.nodes(aGraph)))
		if (random_node2 in aGraph.neighbors(random_node1)):
			aGraph.remove_edge(random_node1,random_node2)
		 	if(nx.is_connected(aGraph)):
		 		k +=1
		 	else :
		 		aGraph.add_edge(random_node1,random_node2)
	return(aGraph)

#Removes a random number of edges and adds a random number of edges in the number_to_change operations the user defined.
def mutation_of_a_graph(aGraph,number_to_change,method = None):
	if method == None :
		a = int(random.random()*number_to_change)
		b = int(number_to_change-a)
		a_changed = 0
		b_changed = 0
		while a_changed < a:
			random_node1 = random.choice(np.array(nx.nodes(aGraph)))
			random_node2 = random.choice(np.array(nx.nodes(aGraph)))
			if (random_node2 in aGraph.neighbors(random_node1)):
				aGraph.remove_edge(random_node1,random_node2)
				a_changed+=1
		while b_changed < b:
			random_node1 = random.choice(np.array(nx.nodes(aGraph)))
			random_node2 = random.choice(np.array(nx.nodes(aGraph)))
			if (random_node2 not in aGraph.neighbors(random_node1)):
				aGraph.add_edge(random_node1,random_node2)
				b_changed+=1
	return(aGraph)

def create_a_population_of_graphs(size_pop,size_graph):
	All_graphs = []
	for graph in range(size_pop):
		All_graphs.append(generate_random_connected_graph(size_graph))
	return (All_graphs)


G = generate_random_connected_graph(100)
max_edges = nx.number_of_edges(G)
#G = remove_edges_to_graph_without_disconnect(G, 300)
#drawhistandgraph(G)
G = mutation_of_a_graph(G,10)
Population = create_a_population_of_graphs(10,10)
print(Population[1])