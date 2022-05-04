zi.pi<-function(nodes_bulk, z.bulk, modularity_class, degree){
  
  z.bulk[abs(z.bulk)>0]<-1
  module<-which(colnames(nodes_bulk)==modularity_class)
  module.max<-max(nodes_bulk[,module])
  degree<-which(colnames(nodes_bulk)==degree)
  bulk.module<-list(NA)
  length(bulk.module)<-module.max
  
  for(i in 1:max(nodes_bulk[,module])){
    bulk.module[[i]]<-z.bulk[which(nodes_bulk[,module]==i),which(nodes_bulk[,module]==i)]
    bulk.module[[i]]<-as.data.frame(bulk.module[[i]])
    rownames(bulk.module[[i]])<-rownames(z.bulk)[which(nodes_bulk[,module]==i)]
    colnames(bulk.module[[i]])<-colnames(z.bulk)[which(nodes_bulk[,module]==i)]
  }
  
  # within-module degree z
  z_bulk<-list(NA)
  length(z_bulk)<-module.max
  
  for(i in 1:length(z_bulk)){
    z_bulk[[i]]<-bulk.module[[i]][,1]
    z_bulk[[i]]<-as.data.frame(z_bulk[[i]])
    colnames(z_bulk[[i]])<-"z"
    rownames(z_bulk[[i]])<-rownames(bulk.module[[i]])
  }
  
  for(i in 1:max(nodes_bulk[,module])){
    if(length(bulk.module[[i]])==1){
      z_bulk[[i]][,1]<-0
    }else if(sum(bulk.module[[i]])==0){
      z_bulk[[i]][,1]<-0
    }else{
      k<-rowSums(bulk.module[[i]]) -1
      mean<-mean(k)
      sd<-sd(k)
      if (sd==0){
        z_bulk[[i]][,1]<-0
      }else{
        z_bulk[[i]][,1]<-(k-mean)/sd
      }
    }
  }
  
  for(i in 2:max(nodes_bulk[,module])) {
    z_bulk[[i]]<-rbind(z_bulk[[i-1]],z_bulk[[i]])
  }
  z_bulk<-z_bulk[[module.max]]
  
  bulk.module1<-list(NA)
  length(bulk.module1)<-module.max
  
  for(i in 1:max(nodes_bulk[,module])){
    bulk.module1[[i]]<-z.bulk[,which(nodes_bulk[,module]==i)]
    bulk.module1[[i]]<-as.data.frame(bulk.module1[[i]])
    rownames(bulk.module1[[i]])<-rownames(z.bulk)
    colnames(bulk.module1[[i]])<-colnames(z.bulk)[which(nodes_bulk[,module]==i)]
  }
  
  #among-module connectivity c
  c_bulk<-list(NA)
  length(c_bulk)<-module.max
  
  for(i in 1:length(c_bulk)){
    c_bulk[[i]]<-z.bulk[,1]
    c_bulk[[i]]<-as.matrix(c_bulk[[i]])
    colnames(c_bulk[[i]])<-"c"
    rownames(c_bulk[[i]])<-rownames(z.bulk)
    c_bulk[[i]][,1]<-NA
  }
  
  for(i in 1:max(nodes_bulk[,module])){
    c_bulk[[i]]<-rowSums(bulk.module1[[i]])
    c_bulk[[i]]<-as.matrix(c_bulk[[i]])
    c_bulk[[i]][which(nodes_bulk$modularity == i),] = c_bulk[[i]][which(nodes_bulk$modularity == i),] -1
    c_bulk[[i]]<-c_bulk[[i]]*c_bulk[[i]]
    colnames(c_bulk[[i]])<-"c"
    rownames(c_bulk[[i]])<-rownames(z.bulk)
  }
  
  for(i in 2:max(nodes_bulk[,module])){
    c_bulk[[i]]<-c_bulk[[i]]+c_bulk[[i-1]]
  }
  c_bulk<-c_bulk[[module.max]]
  
  for(i in 1: length(c_bulk)){
    if(nodes_bulk$degree[i]==0){
      c_bulk[i] <- 0
    }else{
      c_bulk[i] <- 1-(c_bulk[i]/(nodes_bulk$degree[i]*nodes_bulk$degree[i]))
    }
  }
  colnames(c_bulk)<-"c"
  
  z_c_bulk<-c_bulk
  z_c_bulk<-as.data.frame(z_c_bulk)
  z_c_bulk$z<-z_bulk[match(rownames(c_bulk),rownames(z_bulk)),]
  z_c_bulk<-z_c_bulk[,c(2,1)]
  names(z_c_bulk)[1:2]<-c('within_module_connectivities','among_module_connectivities')
  
  z_c_bulk$nodes_id<-rownames(z_c_bulk)
  nodes_bulk$nodes_id<-rownames(nodes_bulk)
  z_c_bulk<-merge(z_c_bulk,nodes_bulk,by='nodes_id')
  z_c_bulk
  
}


library(igraph)
dat<-read.csv('Correlation (0,1).csv', row.names = 1,header = T)
dat[dat > 0] = 1
dat[dat < 0] = 1

adjacency_unweight<-dat
igraph <- graph_from_adjacency_matrix(as.matrix(adjacency_unweight), mode = 'undirected', weighted = NULL, diag = FALSE)
igraph    #igraph 

V(igraph)$degree <- degree(igraph)

#?cluster_fast_greedy
set.seed(123)
V(igraph)$modularity <- membership(cluster_fast_greedy(igraph))

nodes_list <- data.frame(
  nodes_id = V(igraph)$name, 
  degree = V(igraph)$degree, 
  modularity = V(igraph)$modularity
)
head(nodes_list)

write.table(nodes_list, 'nodes_list.txt', sep = '\t', row.names = FALSE, quote = FALSE)

nodes_list <- read.delim('Village-fungi nodes_list.txt', row.names = 1, sep = '\t', check.names = FALSE)

nodes_list <- nodes_list[rownames(adjacency_unweight), ]

zi_pi <- zi.pi(nodes_list, adjacency_unweight, degree = 'degree', modularity_class = 'modularity')
head(zi_pi)

write.table(zi_pi, 'zi_pi_result.txt', sep = '\t', row.names = FALSE, quote = FALSE)

library(ggplot2)

zi_pi <- na.omit(zi_pi) 
zi_pi[which(zi_pi$within_module_connectivities < 2.5 & zi_pi$among_module_connectivities < 0.62),'type'] <- 'Peripherals'
zi_pi[which(zi_pi$within_module_connectivities < 2.5 & zi_pi$among_module_connectivities > 0.62),'type'] <- 'Connectors'
zi_pi[which(zi_pi$within_module_connectivities > 2.5 & zi_pi$among_module_connectivities < 0.62),'type'] <- 'Module hubs'
zi_pi[which(zi_pi$within_module_connectivities > 2.5 & zi_pi$among_module_connectivities > 0.62),'type'] <- 'Network hubs'


ggplot(zi_pi, aes(among_module_connectivities, within_module_connectivities)) +
  geom_point(aes(color = type), alpha = 0.5, size = 2) +
  scale_color_manual(values = c('gray','red','blue','purple'),
                     limits = c('Peripherals', 'Connectors', 'Module hubs', 'Network hubs'))+
  theme(panel.grid = element_blank(), axis.line = element_line(colour = 'black'),
        panel.background = element_blank(), legend.key = element_blank()) +
  labs(x = 'Among-module connectivities', y = 'Within-module connectivities', color = '') +
  geom_vline(xintercept = 0.62) +
  geom_hline(yintercept = 2.5)


