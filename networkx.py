# -*- coding: utf-8 -*-
"""
Created on Sun Mar 04 03:45:34 2018

@author: KevinYeung
"""

import networkx as nx
g = nx.Graph()

bj = "Beijing"
gz = "Guangzhou"
sh = "Shanghai"
wh = "Wuhan"
sz = "Shenzhen"



g.add_edge(bj,sh,weight=1338)
g.add_edge(gz,sh,weight=1801)
g.add_edge(gz,sz,weight=158)
g.add_edge(gz,wh,weight=1046)
g.add_edge(wh,bj,weight=1143)
g.add_edge(bj,gz,weight=2128)


pos = {'Beijing':[int(116.4074),int(39.9042)],
       'Guangzhou':[int(113.2644),int(23.1291)],       
       'Shanghai':[int(121.4737),int(31.2304) ],
        'Wuhan':[int(114.3055),int(30.5928) ],
        'Shenzhen':[ int(114.0579),int(22.5431)],
    }
#pos = nx.spring_layout(g)

nx.draw(g, pos, font_size=16, with_labels=False)

for p in pos:  # raise text positions
    pos[p][1] += 0.5



nx.draw_networkx_labels(g, pos)
plt.show()









