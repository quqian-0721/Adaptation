import numpy as np
import pandas as pd
import numpy as np
import os
import xlrd

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import matplotlib
# fig = plt.figure(figsize=(16, 8),    # 画布尺寸
#                  facecolor='cornsilk'    # 画布背景色
#                 )
   # 海岸线

book = xlrd.open_workbook('map_data_results/results_all_years.xls')
sheet = book.sheet_by_index(0)
rows = sheet.nrows
cols = sheet.ncols
features_index=[]
all_sites_lon=[]
all_sites_lat=[]
gbr_values=[]
rf_values=[]
xgb_values=[]
for i in range(1,rows):
    all_sites_lon.append(float(sheet.cell(i,2).value))
    all_sites_lat.append(float(sheet.cell(i,3).value))
    gbr_values.append(float(sheet.cell(i,cols-3).value))
    rf_values.append(float(sheet.cell(i,cols-2).value))
    xgb_values.append(float(sheet.cell(i,cols-1).value))
# m.drawcountries()    # 国界线

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111)
# 绘制地图
# m = Basemap(projection='lcc',    # 投影类型
#             llcrnrlon=77, llcrnrlat=14,    # 左下角经纬的
#             urcrnrlon=148, urcrnrlat=57,    # 右上角经纬度
#             lat_1=33, lat_2=45, lon_0=100    # 
#            )

cm = plt.cm.get_cmap('RdYlBu_r')
m = Basemap()    # 投影类型
m.drawcoastlines() 
parallels = np.arange(-90., 120., 30.)  # 这两行画纬度，范围为[-90,90]间隔为10
# m.drawparallels(parallels,labels=[False, True, True, False])
meridians = np.arange(-180., 180., 60.)  # 这两行画经度，范围为[-180,180]间隔为10
# m.drawmeridians(meridians,labels=[True, False, False, True])

m.drawparallels(parallels,labels=[True,False,False,False],linewidth=1,dashes=[1,10], fontsize=20)##纬度


m.drawmeridians(meridians,labels=[False,False,False,True],linewidth=1,dashes=[1,10], fontsize=20) ##jing度

# lon, lat为给定的经纬度，可以使单个的，也可以是列表
plot=m.scatter(all_sites_lon, all_sites_lat, c=gbr_values,s=20, cmap=cm,vmin=-0.3, vmax=0.3)
# plot=m.scatter(all_sites_lon, all_sites_lat, c=gbr_values,s=20, alpha=0.4,cmap=matplotlib.cm.seismic,vmin=-0.5, vmax=0.3)

# m.scatter(all_sites_lon, all_sites_lat, c=gbr_values,cmap=cm)
# plt.clim(-1,1)
# cores=ax.set_clim(-1, 1)
cb=plt.colorbar(plot)
cb.ax.tick_params(labelsize=20)
# cbar_ticks = np.linspace(-1., 1., num=5, endpoint=True)
# cb.set_ticks(cbar_ticks)

plt.savefig("map_data_results/gbr_all_years.pdf",dpi = 600)
plt.show()
