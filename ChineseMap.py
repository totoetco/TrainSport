# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 08:15:47 2018

@author: yyang1
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


bj = (39.9042, 116.4074)

plt.figure(figsize=(16,8))

m = Basemap(llcrnrlon=77, llcrnrlat=14, urcrnrlon=140, urcrnrlat=51, projection='lcc', lat_1=33, lat_2=45, lon_0=100)
m.drawcoastlines()
m.drawcountries(linewidth=1.5) #draw country lines
m.scatter(bj[1],bj[0],latlon=True)# to draw a specitic city

plt.show()