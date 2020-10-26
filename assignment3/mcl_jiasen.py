import numpy as np
import markov_clustering as mc
import networkx as nx
import pandas as pd

file=pd.read_excel(open('D:/LEARN/GRADUATE2020/IDA/assignment3/mcl.xlsx','rb'),sheet_name='Sheet3')
matrix=np.array(file)
g=nx.Graph()
innum=2.1 #change inflation values
for i in matrix:
    g.add_edge(i[0],i[1],weight=i[2])
a=nx.to_numpy_array(g)
result=mc.run_mcl(a,inflation=innum)
for j in range(9):
    result=mc.run_mcl(result,inflation=innum)

clusters=mc.get_clusters(result)
#title=[('A',float),('B',float),('C',float),('D',float),('E',float),('F',float),('G',float),('H',float),('J',float),('K',float),('L',float),('M',float),('N',float),('P',float),('Q',float),('R',float),('S',float)]
mc.draw_graph(result,clusters,with_labels=True)
