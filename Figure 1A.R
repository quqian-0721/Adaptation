library(ggspatial)
library(sf)
library(ggplot2)
library(tidyverse)
china_shp <- "中国省级地图GS（2019）1719号.geojson"
nine <- "九段线GS（2019）1719号.geojson"
china <- sf::read_sf(china_shp)
nine_line <- sf::read_sf(nine)
library(ggspatial)
ggplot() + 
  geom_sf(data = china,fill="NA",size=1,color="black") + 
  geom_sf(data = nine_line) + 
  coord_sf(crs = "+proj=laea +lat_0=40 +lon_0=104")+
  annotation_scale(location = "bl") +
  # spatial-aware automagic north arrow
  annotation_north_arrow(location = "tl", which_north = "false",
                         style = north_arrow_fancy_orienteering)
mydata<-read.table("Longitude and Latitude.txt",head=TRUE, stringsAsFactors=FALSE)
scatter_df_tro <- st_as_sf(mydata,coords = c("Lon", "Lat"),crs = 4326)
library(cowplot)

ggplot() + 
  geom_sf(data = china,fill=NA) + 
  geom_sf(data = nine_line,color='gray50',size=.8)+
  geom_sf(data = scatter_df_tro,aes(fill=Class,size=3,shape=Class))+
  coord_sf(ylim = c(-2387082,1654989),crs="+proj=laea +lat_0=40 +lon_0=104")+
  scale_fill_manual(values = c("#E21C21","#3A7CB5","#51AE4F","#f58220"))+
  scale_size(range = c(1,5))+
  scale_shape_manual(values = c(21,22,23,24))+
  annotation_scale(location = "bl",text_face = "bold",
                   text_family = "Times_New_Roman") +
  # spatial-aware automagic north arrow
  annotation_north_arrow(location = "tl", which_north = "false",
                         style = north_arrow_fancy_orienteering,
  )+
  guides(fill = guide_legend(override.aes = list(size = 3),
                             title = "",
                             label.position = "right",
                             ncol=3,
  ),
  size = guide_legend(
    title = "",
    label.position = "right",
    ncol=5)) +
  #theme_bw()+
  theme(
    text = element_text(family = 'Times_New_Roman',size = 18,face = "bold"),
    
    panel.background = element_rect(fill = NA),
    panel.grid.major = element_line(colour = "grey80",size=.2),
    legend.key = element_rect(fill = "white"),
    legend.position = "bottom",
  )
#九段线
ggplot() +
  geom_sf(data = china,fill='NA') + 
  geom_sf(data = nine_line,color='gray70',size=1.)+
  #geom_sf(data = scatter_df_tro,aes(fill=class,size=data),shape=21,colour='black',stroke=.25)+
  coord_sf(ylim = c(-4028017,-1877844),xlim = c(117131.4,2115095),crs="+proj=laea +lat_0=40 +lon_0=104")+
  theme(
    #aspect.ratio = 1.25, #调节长宽比
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    panel.background = element_blank(),
    panel.border = element_rect(fill=NA,color="grey10",linetype=1,size=1.),
    plot.margin=unit(c(0,0,0,0),"mm"))


gg_inset_map = ggdraw() +
  draw_plot(map) +
  draw_plot(nine_map, x = 0.8, y = 0.15, width = 0.1, height = 0.3)

