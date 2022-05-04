library(vegan)
library(ape)
library(ggplot2)
library(ggrepel)

data <- read.csv("otu.txt", head=TRUE,sep="\t",row.names = 1)
groups <- read.table("group.txt",sep = "\t",header = F,colClasses = c("character"))
groups <- as.list(groups)
data <- t(data)
data[is.na(data)] <- 0
data <- vegdist(data,method = "bray")
pcoa<- pcoa(data, correction = "none", rn = NULL)
PC1 = pcoa$vectors[,1]
PC2 = pcoa$vectors[,2]
plotdata <- data.frame(rownames(pcoa$vectors),PC1,PC2,groups$V2)
colnames(plotdata) <-c("sample","PC1","PC2","group")
pich=c(21:24)
cbbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442")
Palette <- c("#000000", "#000000", "#000000", "#000000")
pc1 <-floor(pcoa$values$Relative_eig[1]*100)
pc2 <-floor(pcoa$values$Relative_eig[2]*100)

otu.adonis=adonis(data~V2,data = groups,distance = "bray")

ggplot(plotdata, aes(PC1, PC2)) +
  geom_point(aes(colour=group,shape=group,fill=group),size=12)+
  geom_text(aes(x = 0.05,y = 0.35,label = paste("PERMANOVA:\n    Group_A VS Group_B\n    p-value = ",otu.adonis$aov.tab$`Pr(>F)`[1],sep = "")),size = 10,hjust = 0)+
  stat_ellipse(aes(fill = group),geom = "polygon",level = 0.95,alpha = 0.3)+
  scale_shape_manual(values=pich)+
  scale_colour_manual(values=Palette)+
  scale_fill_manual(values=cbbPalette)+
  labs(title="PCoA - The composition of gut microbiome") + 
  xlab(paste("PC1 ( ",pc1,"%"," )",sep="")) + 
  ylab(paste("PC2 ( ",pc2,"%"," )",sep=""))+
  theme(text=element_text(size=30))+
  geom_vline(aes(xintercept = 0),linetype="dotted")+
  geom_hline(aes(yintercept = 0),linetype="dotted")+
  theme(panel.background = element_rect(fill='white', colour='black'),
        panel.grid=element_blank(), 
        axis.title = element_text(color='black',size=34),
        axis.ticks.length = unit(0.4,"lines"), axis.ticks = element_line(color='black'),
        axis.line = element_line(colour = "black"), 
        axis.title.x=element_text(colour='black', size=34),
        axis.title.y=element_text(colour='black', size=34),
        axis.text=element_text(colour='black',size=28),
        legend.title=element_blank(),
        legend.text=element_text(size=34),
        legend.key=element_blank(),legend.position = c(0.12,0.88),
        legend.background = element_rect(colour = "black"),
        legend.key.height=unit(1.6,"cm"))+
  theme(plot.title = element_text(size=34,colour = "black",hjust = 0.5,face = "bold"))



