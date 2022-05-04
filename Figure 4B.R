rm(list=ls())
otu <- read.table("NST1.txt", header = T, row.names = 1)

counts<-rowSums(otu>0)
otu<-otu[counts>=105,]#Be sure that each OTU appears in at least five samples
otu <- data.frame(t(otu))


library(ape)
tree <- read.tree('otu_phylo.tre')
library(picante)
tree <- prune.sample(otu, tree)
dis <- cophenetic(tree)

set.seed(123)
mntd <- ses.mntd(otu, dis, abundance.weighted = TRUE, null.model = 'taxa.labels', runs = 999)
mntd
#The negative value of mntd.obs.z is NTI
mntd$NTI <- mntd$mntd.obs.z * -1
mntd

library(NST)
library(picante)
library(ape)

#Read the OTU abundance table
otu <- read.table("NST1.txt", header = T, row.names = 1)

counts<-rowSums(otu>0)
otu<-otu[counts>=105,]
otu <- data.frame(t(otu))

group <- read.table("NSTgroup.txt", header = T, row.names = 1)

tree <- read.tree('otu_phylo.tre')


tree <- prune.sample(otu, tree)

set.seed(123)
pnst <- pNST(comm = otu, tree = tree, group = group, phylo.shuffle = TRUE, taxo.null.model = NULL, 
             pd.wd = tempdir(), abundance.weighted = TRUE, rand = 1000, nworker = 4, SES = TRUE, RC = FALSE)

betaMNTD <- pnst$index.pair
head(betaMNTD)

FSA <- rownames(subset(group, treatment=='FSA'))
betaMNTD_FSA <- subset(betaMNTD, name1 %in% FSA & name2 %in% FSA)
SSA <- rownames(subset(group, treatment=='SSA'))
betaMNTD_SSA <- subset(betaMNTD, name1 %in% SSA & name2 %in% SSA)
FSC <- rownames(subset(group, treatment=='FSC'))
betaMNTD_FSC <- subset(betaMNTD, name1 %in% FSC & name2 %in% FSC)
SSC <- rownames(subset(group, treatment=='SSC'))
betaMNTD_SSC <- subset(betaMNTD, name1 %in% SSC & name2 %in% SSC)

#|¦ÂNTI|<2
nrow(betaMNTD_FSA[which(abs(betaMNTD_FSA$bNTI.wt)<2), ])/nrow(betaMNTD_FSA)  
nrow(betaMNTD_SSA[which(abs(betaMNTD_SSA$bNTI.wt)<2), ])/nrow(betaMNTD_SSA)  
nrow(betaMNTD_FSC[which(abs(betaMNTD_FSC$bNTI.wt)<2), ])/nrow(betaMNTD_FSC)  
nrow(betaMNTD_SSC[which(abs(betaMNTD_SSC$bNTI.wt)<2), ])/nrow(betaMNTD_SSC) 

#|¦ÂNTI|>2
nrow(betaMNTD_FSA[which(abs(betaMNTD_FSA$bNTI.wt)>2), ])/nrow(betaMNTD_FSA) 
nrow(betaMNTD_SSA[which(abs(betaMNTD_SSA$bNTI.wt)>2), ])/nrow(betaMNTD_SSA)  
nrow(betaMNTD_FSC[which(abs(betaMNTD_FSC$bNTI.wt)>2), ])/nrow(betaMNTD_FSC)  
nrow(betaMNTD_SSC[which(abs(betaMNTD_SSC$bNTI.wt)>2), ])/nrow(betaMNTD_SSC)  

write.csv(betaMNTD_SSC, "SSC.csv")

library(ggpubr)

betaMNTD_FSA$group <- 'FSA'
betaMNTD_SSA$group <- 'SSA'
betaMNTD_FSC$group <- 'FSC'
betaMNTD_SSC$group <- 'SSC'
betaMNTD_group <- rbind(betaMNTD_FSA, betaMNTD_SSA,betaMNTD_FSC, betaMNTD_SSC)

ggboxplot(data = betaMNTD_group, x = 'group', y = 'bNTI.wt', color = 'group') +
  scale_x_discrete( limits=c("FSA","SSC","FSC","SSA"))+
  scale_color_manual(values=c("#be9c2e","#fe832d","#c72e29","#016392"))+
  geom_hline (yintercept =  c (-2,2),colour="black",linetype="dashed", size=0.8)+
  labs(y = 'betaNTI')
