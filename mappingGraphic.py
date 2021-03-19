# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 05:35:32 2020

@author: Berk
"""
import folium
import pandas as pd

data = pd.read_excel('D:\Documents\Spyder\Deep Learning\datas\YururkenVeri.xlsx')
datagps = pd.read_excel('D:\Documents\WorkStation\Bitirme Projesi\datas\gpsverisi.xlsx')
Y = datagps['Enlem'].values.tolist()
Z = datagps['Boylam'].values.tolist()

#40.7514794,30.3494019 verilerin alındığı mekanın lokasyon bilgisi.

folium_map = folium.Map(location=[40.7514794,30.3494019],
                        zoom_start=13,
                        tiles="CartoDB dark_matter")
for i in range (1,150):
    marker = folium.CircleMarker(location=[Y[i],Z[i]]) #GPS verilerini yerleştir.
    marker.add_to(folium_map)

folium_map.save("my_map.html")