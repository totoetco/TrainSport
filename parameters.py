import numpy as np
import powerlaw
import networkx as nx
import matplotlib.pyplot as plt

### Fonctions for the Leven-Marquart algorithm ###

# cost function
def f(x,y,a):
	return(0.5*np.sum((y-power(x,a))**2))

#gradient function
def gradient(x,y,a):
	return(-np.sum((y-power(x,a))*(np.log(x)*power(x,a))))

# Hessienne function
def deriv_seconde(x,a):
	return(np.sum((np.log(x)*power(x,a)**2)))

# Algorithme de Leven-Marquart
def LM (a, l, f, g, ds , x, y):
	
	grad = gradient(x,y,a)
	it = 0
	while grad > 10**(-5) or it>1000:
		grad = gradient(x,y,a)
		hessienne = ds(x,a)*(1+l)
		d = -float(grad)/hessienne
		cout = f(x, y, a)
		if f(x, y, a+d)<cout:
			a = a+d
			l = l/10
		else: 
			l = l*10
		it+=1
	return(a)

# power law function
def power(x, p):
	return (x**p)

# Returns the degree distribution of a graph
def deg_distribution(G):


	d = sorted(d for n, d in G.degree())
	x = []
	y = []

	for deg in d:
		if deg not in x:
			x.append(deg)
			y.append(d.count(deg))

	y=np.asarray(y)/max(y)

	#plt.bar(x,y)
	#plt.title("Degree distribution.")
	#plt.show()
	return(np.asarray([x,y]))

# Returns the degree coefficient for a given graph
def degree_coef(G):
	
	d=deg_distribution(G)

	return(-LM(0, 0.001, f, gradient, deriv_seconde, d[0], d[1]))

# Plot the clustering coefficient distribution and computes the clustering power law coefficient
def clustering_coef(G):

	clus=nx.clustering(G)
	
	h=plt.hist(clus.values(), bins=100)
	#plt.title("Clustering coefficient distribution.")
	#plt.show()

	y=h[0]/max(h[0])
	x=h[1][0:100]
	x[0]=0.001
	#plt.plot(x,y, 0.001)
	#plt.title("Clustering coefficient scaled distribution.")
	#plt.show()

	return(-LM(0, 0.001, f, gradient, deriv_seconde, x,y))



#### power law fitting function: Test data!! ######
#x = np.linspace(1 ,5, 100)
#y = power(x, -2.5) + 0.2*np.random.normal(0,1, 100)

#print(x, y)
#plt.plot(x, power(x, -2.5))
#plt.plot(x, y, 'x')
#plt.title("Test for power-law fitting algorthm on fake data!")
#plt.show()

#res = LM(0, 0.001, f, gradient, deriv_seconde, x, y)




# Creates a barabasi random graph
G = nx.barabasi_albert_graph(350, 100)
print('The graph contains', len(list(G.edges())), ' edges.')
print('The graph contains ', len(list(G.nodes())), ' nodes.')

# Plot the graph (careful with vast number of nodes)
#plt.subplot(111)
#nx.draw(G, with_labels=True, font_weight='bold')
#plt.show()

# Show the parameters of the graph
#gamma = degree_coef(G)
#print('gamma: ', gamma, '\n')
#beta = clustering_coef(G)
#C = nx.average_clustering(G)
#print('C: ', C, '\nbeta: ', beta, '\n')
#L = nx.average_shortest_path_length(G)
#print('L: ', L, '\n')

