
library(ggplot2)
library(tabplot)
#devtools::install_github("mtennekes/tabplot")
library(RColorBrewer) 
diamonds<-read.csv("xx.csv",row.names = 1,header = T)
tabplot::tableplot(diamonds, sortCol =price, 
                   select = c(price,carat,cut, color, clarity,depth, table),
                   pals = list(cut=palette(brewer.pal(9,"Purples"))[c(2,3,4,5,7,8)],
                               color=palette(brewer.pal(9,"Oranges"))[c(2:9)], 
                               clarity=palette(brewer.pal(9,"Greens"))[1:8]),
                   fontsize = 9)
