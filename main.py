#!/usr/bin/env python
import networkx as nx
import os.path
import numpy
import pdb
import sys
import math
import collections
from collections import Counter
from random import choice
import random 
from scipy.optimize import minimize
import time
import numpy.linalg as LA


algorithm = None # the algorithm created as global variable
DIMENSION = 4
SAMPLE_SIZE = 30
#SHARP = 100000.0
SHARP = 50.0
LAMBDA1 = 0.7 
LAMBDA2 = 100.0 / SAMPLE_SIZE
AVE = 1  
#LAMBDA2 = 80000.0 / SAMPLE_SIZE



def f_wrapper(p):
  return algorithm.f(p)

def f_der_wrapper(p):
  return algorithm.f_der(p)

def solve():
  global algorithm
  starttime = time.time()
  res = minimize(f_wrapper, algorithm.p, method="BFGS", jac=f_der_wrapper, \
				options={'maxiter': 8, 'gtol': 1e-4, 'disp': True})
  endtime = time.time()
  print "time of convergence:",
  print(endtime - starttime)
  print "final result: p=", (res.x)

def compute_psis(N,t):
	psis = {}
	psis[N] = 1.
	for i in xrange(N-1,0,-1):
		psis[i] = psis[i+1]*t/(float(i+1.))+1.
	return psis

def f1score(nodes, heatnodes):
	common = len(nodes.intersection(heatnodes))
	precision = 1.0 * common / len(nodes)
	recall = 1.0 * common / len(heatnodes)
	if precision + recall == 0:
		return 0
	else:
		return ( 2.0 * precision * recall / (precision + recall) )

def sigmoid(x):
	#return max(x, 0.0)
	x = SHARP * x
	if x > 10: x = 10.0 
	if x < -10: x = -10.0
	return 1.0 / ( 1.0 + math.exp(- x) ) 

def sigmoid_der(x):
	#if x <= 0:
	#	 return 0.0
	#else:
	#	 return 1.0
	return SHARP * sigmoid(x) * ( 1.0 - sigmoid(x) ) 

class AdaptiveModularity:
	def __init__(self):
		self.loadFootball()

	def loadAmazon(self):
		if os.path.isfile("./amazon.gpickle"):
			self.G = nx.read_gpickle("./amazon.gpickle")
		else:
			self.G = nx.Graph(gnc = {}, membership = {}, top5000 = {})
			with open("com-amazon.ungraph.txt", "r") as txt:
				for line in txt:
					if not line[0] == '#':
						e = line.split()
						self.G.add_edge(int(e[0]), int(e[1]))
			with open("com-amazon.top5000.cmty.txt", "r") as txt:
				count = 0
				for line in txt:
					if not line[0] == '#':
						e = line.split()
						self.G.graph["top5000"][count] = [int(_) for _ in e]   
						count += 1
			with open("com-amazon.all.dedup.cmty.txt", "r") as txt:
				count = 0
				for line in txt:
					if not line[0] == '#':
						e = line.split()
						self.G.graph["gnc"][count] = [int(_) for _ in e]   
						for n in self.G.graph["gnc"][count]:
							if n in self.G.graph["membership"]:
								self.G.graph["membership"][n].append( count )
							else:
								self.G.graph["membership"][n] = [ count ]
						count += 1
			print "write gpickle file.."
			nx.write_gpickle(self.G,"./amazon.gpickle")

	def loadFootball(self):
		self.G = nx.Graph(gnc = {}, membership = {})
		with open("./football_data/footballTSEinputEL.dat", "r") as f:
			for line in f:
				seg = line.split()
				self.G.add_edge( int(seg[0]), int(seg[1]) )
		with open("./football_data/footballTSEinputConference.clu", "r") as fconf:
			for i, line in enumerate(fconf):
				self.G.graph["membership"][i] = [ str(line.strip()) ]
				if str(line.strip()) in self.G.graph["gnc"]:
					self.G.graph["gnc"][str(line.strip())].append(i) 
				else:
					self.G.graph["gnc"][str(line.strip())] = [ i ]
		print "write gpickle file.."
		nx.write_gpickle(self.G,"./football.gpickle")
		#with open("./football_weight.group", "w+") as txt:
		#	for k, v in self.G.graph["gnc"].items():
		#		txt.write("\t".join([str(_) for _ in v]))
		#		txt.write("\n")

	def edge_feature(self, e):
		return numpy.array([ math.sqrt(float(len(set(self.G[e[0]]).intersection(self.G[e[1]])))), \
							 float(abs(nx.clustering(self.G, e[0]) - nx.clustering(self.G, e[1]))), \
							 float(list(nx.jaccard_coefficient(self.G, [(e[0], e[1])]))[0][2]), \
							 1.0
						   ])
							 #float(min(len(self.G[e[0]]), len(self.G[e[1]]))), \
							 #float(max(len(self.G[e[0]]), len(self.G[e[1]]))), \
							 #float(list(nx.resource_allocation_index(self.G, [(e[0], e[1])]))[0][2]), \
							 #float(list(nx.preferential_attachment(self.G, [(e[0], e[1])]))[0][2]), \
							 #float(list(nx.adamic_adar_index(self.G, [(e[0], e[1])]))[0][2]), \

	def preprocess(self):
		self.dimension = DIMENSION 
		self.numiter = 0 
		self.p = numpy.ones(self.dimension)
		self.edges_involved = set() 
		self.comm = {}
		self.pairs = set() 
		for t in range(SAMPLE_SIZE):
			i = choice(self.G.graph["gnc"].keys())
			nodes_i = self.G.graph["gnc"][i]
			if len(nodes_i) > 10 and len(nodes_i) < 100: 
				self.preprocess_helper(i, nodes_i)
				memberships = [self.G.graph["membership"][e[1]]
								for e in self.comm[i]["eout"]
								if e[1] in self.G.graph["membership"] 
							] 
				if not memberships: continue
				neighbor_comms = reduce(lambda x,y: list(x+y), memberships)
				for _ in range(min(len(neighbor_comms), 30)):
					j = choice(neighbor_comms)
					nodes_j = self.G.graph["gnc"][j]
					if len(nodes_j) > 5 and len(nodes_j) < 100 and f1score(set(nodes_i), set(nodes_j)) < 0.4 :
						print "comm",i,"(",len(nodes_i),"nodes)",": comm",j,"(",len(nodes_j),"nodes)"		 
						self.preprocess_helper(j, nodes_j)
						self.preprocess_helper((i,j), list(set(nodes_i + nodes_j)))
						self.pairs.add((i,j)) 
						break

		self.ave_edge_feature = 1.0 * numpy.sum([ self.G.edge[e[0]][e[1]]["feature"] for e in self.edges_involved], axis=0) \
										/ float(len(self.edges_involved))
		self.sum_edge_feature = self.ave_edge_feature * self.G.number_of_edges()

		self.cov_edge_feature = numpy.zeros((self.dimension, self.dimension)) 
		for e in self.edges_involved:
			self.cov_edge_feature = numpy.add(self.cov_edge_feature,\
											numpy.dot(\
												self.G.edge[e[0]][e[1]]["feature"][numpy.newaxis].T,\
												self.G.edge[e[0]][e[1]]["feature"][numpy.newaxis])\
										)
		self.cov_edge_feature /= float(len(self.edges_involved))
		print "covariance matrix", self.cov_edge_feature
		print "Average edge feature:", self.ave_edge_feature
		return

	def preprocess_helper(self, i, nodes): 
		self.comm[i] = {}

		edges = self.G.edges(nodes)
		for e in edges:
			self.G.edge[e[0]][e[1]]["feature"] = self.edge_feature(e) 
			self.edges_involved.add(e) # used for statistics

		self.comm[i]["ein"] = [e for e in edges if e[0] in nodes and e[1] in nodes]
		self.comm[i]["eout"] = [e for e in edges if not e in self.comm[i]["ein"]]

		self.comm[i]["in"] = numpy.sum([ self.G.edge[e[0]][e[1]]["feature"] for e in self.comm[i]["ein"] ], axis=0)
		self.comm[i]["cut"] = numpy.sum([ self.G.edge[e[0]][e[1]]["feature"] for e in self.comm[i]["eout"] ], axis=0)
		self.comm[i]["vol"] = 2.0 * self.comm[i]["in"] + self.comm[i]["cut"] # vol = 2 ein + eout

	def update(self, p, i, j):
		vol_i_j = numpy.inner(p, self.comm[(i,j)]["vol"])
		cut_i_j = numpy.inner(p, self.comm[(i,j)]["cut"])
		in_i_j = numpy.inner(p, self.comm[(i,j)]["in"])
		 
		vol_i = numpy.inner(p, self.comm[i]["vol"])
		cut_i = numpy.inner(p, self.comm[i]["cut"])
		in_i = numpy.inner(p, self.comm[i]["in"])

		vol_j = numpy.inner(p, self.comm[j]["vol"])
		cut_j = numpy.inner(p, self.comm[j]["cut"])
		in_j = numpy.inner(p, self.comm[j]["in"])
		return vol_i_j, cut_i_j, in_i_j, vol_i, cut_i, in_i, vol_j, cut_j, in_j

	def f(self, p):
		'''
		@return the objective function 
		'''
		ret = 0
		total_negative = 0
		self.p = p
		print "p = ", p 
		ave_w = numpy.inner(p, self.ave_edge_feature)
		print "ave_w = ", ave_w
		W = ave_w * self.G.number_of_edges()
		for i,j in self.pairs:
			vol_i_j, cut_i_j, in_i_j, vol_i, cut_i, in_i, vol_j, cut_j, in_j = self.update(p, i, j)
			m_i = (in_i)/W - (vol_i / 2.0 / W) ** 2.0
			m_j = (in_j)/W - (vol_j / 2.0 / W) ** 2.0
			m_i_j = (in_i_j)/W - (vol_i_j / 2.0 / W) ** 2.0
			if vol_i < 0 or vol_j < 0:
				total_negative += 1
			#ret += max( m_i_j - m_i - m_j , 0 ) ** 2.0 
			ret += sigmoid( m_i_j - m_i - m_j ) 
		#ret = numpy.inner(self.p, self.p) + LAMBDA1 * (ave_w - 1.0) ** 2.0  +	LAMBDA2 * ret
		#ret = LAMBDA1 * (ave_w - AVE) ** 2.0  +  LAMBDA2 * ret
		var = numpy.dot( p, numpy.dot(p, self.cov_edge_feature) ) - ave_w ** 2.0
		print "f=" , (ave_w - AVE) ** 2.0 + LAMBDA1 * (var) , "+" ,  LAMBDA2 * ret,
		ret = (ave_w - AVE) ** 2.0 + LAMBDA1 * ( var )	+  LAMBDA2 * ret
		print "==" , ret 

		#should comment this line
		print "negative", total_negative / float(len(self.pairs))  
		print "var", var 
		self.validate()

		return ret

	def f_der(self, p):
		'''
		@return the first order derivetive of the objective function 
		'''
		if self.numiter >= 10: return numpy.zeros(self.dimension)
		self.numiter += 1
		ret = 0
		self.p = p
		ave_w = numpy.inner(p, self.ave_edge_feature)
		W = ave_w * self.G.number_of_edges()
		for i,j in self.pairs:
			vol_i_j, cut_i_j, in_i_j, vol_i, cut_i, in_i, vol_j, cut_j, in_j = self.update(p, i, j)
			m_i = (in_i)/W - (vol_i / 2.0 / W) ** 2.0
			m_j = (in_j)/W - (vol_j / 2.0 / W) ** 2.0
			m_i_j = (in_i_j)/W - (vol_i_j / 2.0 / W) ** 2.0
			#print m_i_j - m_i - m_j
			ret += sigmoid_der( m_i_j - m_i - m_j )  * \
					 ( 
					   ( (self.comm[(i,j)]["in"] * W -	in_i_j * self.sum_edge_feature)  \
					   - 0.5 * vol_i_j / W * (self.comm[(i,j)]["vol"] * W - vol_i_j * self.sum_edge_feature) )\
					   / (W ** 2.0)\
					 - ( (self.comm[i]["in"] * W -	in_i * self.sum_edge_feature)  \
					   - 0.5 * vol_i / W * (self.comm[i]["vol"] * W - vol_i * self.sum_edge_feature) )\
					   / (W ** 2.0)\
					 - ( (self.comm[j]["in"] * W -	in_j * self.sum_edge_feature)  \
					   - 0.5 * vol_j / W * (self.comm[j]["vol"] * W - vol_j * self.sum_edge_feature) )\
					   / (W ** 2.0)\
					 )
		#ret = 2.0 * self.p + 2.0 * LAMBDA1 * (ave_w - 1.0) * self.ave_edge_feature  + 2.0 * LAMBDA2 * ret
		#ret = 2.0 * LAMBDA1 * (ave_w - AVE) * self.ave_edge_feature  + 2.0 * LAMBDA2 * ret
		ret = 2.0 * (ave_w - AVE) * self.ave_edge_feature + 2.0 * LAMBDA1 * ( numpy.dot(p, self.cov_edge_feature) - ave_w * self.ave_edge_feature ) * self.ave_edge_feature  + 2.0 * LAMBDA2 * ret
		#print "f_der =", ret
		return ret

	def gephi(self, i, j, nodes1, nodes2):
		nodes = list(set(nodes1 + nodes2))
		neighbors = reduce(lambda x,y:x + y, \
						[list(set(self.G.neighbors(_)) - set(nodes)) for _ in nodes])
		neighbors2 = reduce(lambda x,y:x + y, \
			[list(set(self.G.neighbors(_)) - set(nodes)) for _ in neighbors])
		#for debug
		neighbors2 = [] 
		neighbors = list(set(neighbors2 + neighbors))

		subG = nx.subgraph(self.G, neighbors + nodes)
		for neighbor in neighbors:
			subG.node[neighbor]["label"] = 0 
		for node in nodes1:
			subG.node[node]["label"] = 1 
		for node in nodes2:
			subG.node[node]["label"] = 2 
		for node in set(nodes1).intersection(set(nodes2)):
			subG.node[node]["label"] = 12 
		with open("./tmpfiles/subg%d_%d.gdf" % (i, j), "w+") as txt:
			txt.write("nodedef>name VARCHAR,label VARCHAR\n")
			for n in subG.nodes():
				txt.write("%d,%d\n" % (n,subG.node[n]["label"]))
			txt.write("edgedef>node1 VARCHAR,node2 VARCHAR,weight VARCHAR\n")
			for e in subG.edges():
				self.G.edge[e[0]][e[1]]["feature"] = self.edge_feature(e)
				self.G.edge[e[0]][e[1]]["weight"] = numpy.inner(self.p, self.G.edge[e[0]][e[1]]["feature"])
				txt.write("%d,%d,%f\n" %( e[0], e[1], self.G.edge[e[0]][e[1]]["weight"]))

	def validate(self):
		p = self.p
		count = 0
		good = 0
		W = float( self.G.number_of_edges() )
		for i,j in self.pairs:
			vol_i_j, cut_i_j, in_i_j, vol_i, cut_i, in_i, vol_j, cut_j, in_j = self.update(p, i, j)
			m_i = (in_i)/W - (vol_i / 2.0 / W) ** 2.0
			m_j = (in_j)/W - (vol_j / 2.0 / W) ** 2.0
			m_i_j = (in_i_j)/W - (vol_i_j / 2.0 / W) ** 2.0
			#print "i", "j", m_i, m_j, m_i_j
			#if m_i_j < m_i + m_j:
			#	good += 1
			#count += 1
		print "=" * 20
		print "p=", self.p
		#print "Correctness : ", ( 100.0 * good / count )
		print "=" * 20

	def draw(self):
		for _ in range(100):
			for i,j in random.sample(self.pairs, 10):
				nodes_i = self.G.graph["gnc"][i]
				nodes_j = self.G.graph["gnc"][j]
				self.gephi(i, j, nodes_i, nodes_j) 

	def fmeasure(self):
		stats = []
		for i,nodes in self.G.graph["top5000"].items():
			f1score_list = []
			for comm in set(reduce(lambda x,y:x+y, [self.G.graph["membership"][n] for n in nodes])):
				thenodes = self.G.graph["gnc"][comm]
				f1score_list.append( f1score(set(thenodes), set(nodes)) )
			print max(f1score_list)
			stats.append(max(f1score_list)) 
		print 1.0 * sum(stats)/len(stats) 

	def convert(self, filename):
		print "total #edges to write:", self.G.number_of_edges()
		with open(filename, "w+") as txt:
			for e in self.G.edges():
				txt.write("%d %d %f\n" % (e[0], e[1], numpy.inner(self.p, self.edge_feature(e))))


if __name__ == "__main__":
	global algorithm
	print "start"
	algorithm = AdaptiveModularity()

	print "preprocessing graph"
	algorithm.preprocess()

	algorithm.p = numpy.zeros(algorithm.dimension)
	algorithm.p[-1] = 1
	algorithm.validate()
	solve()
	algorithm.validate()

	print "output weighted graph"
	algorithm.convert("football.wpairs")

	#algorithm.convert()
	#print "draw sample community pairs"
	#algorithm.draw()
