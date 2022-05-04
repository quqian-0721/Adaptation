library(vegan)
setwd("C:/Users/曲乾/Desktop/Reviewer#1/Response3")
#读取上述数据集
df1 <- read.table('Stage1-Bray-SSA.txt', header= TRUE,row.names = 1)
df2 <- read.table('Stage2-Bray-SSA.txt', header= TRUE,row.names = 1)
df3 <- read.table('Stage3-Bray-SSA.txt', header= TRUE,row.names = 1)
##计算距离
#根据物种丰度数据，计算样方间的 Bray-curtis 距离，数字为OTUs所在的列数
dist.abund1 <- vegdist(df1, method = 'bray')
dist.abund2 <- vegdist(df2, method = 'bray')
dist.abund3 <- vegdist(df3, method = 'bray')
dis1<-as.matrix(dist.abund1)
dis2<-as.matrix(dist.abund2)
dis3<-as.matrix(dist.abund3)
dis_env1 <- as.vector(as.dist(dis1))
dis_env2 <- as.vector(as.dist(dis2))
dis_env3 <- as.vector(as.dist(dis3))
dat <- data.frame(
  dis = c(dis_env1, dis_env2,dis_env3),
  group = factor(c(
    rep('stage1', length(dis_env1)), 
    rep('stage2', length(dis_env2)),
    rep('stage3', length(dis_env3)))
  ), levels = c('stage1', 'stage1','stage3'))

library(ggplot2)

p <- ggplot(dat, aes(group, dis)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values = c('#CD5B45', '#228B22', '#00688B')) +
  theme(panel.grid = element_blank(), panel.background = element_blank(), 
        axis.line = element_line(colour = 'black'), legend.position = 'none') +
  labs(x = NULL, y = 'Bray-Curtis dissimilarity\n')

p
write.csv(dat, "dat-SSA.csv")
