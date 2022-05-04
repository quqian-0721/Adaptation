library(WGCNA)
library("multtest")
library(reshape2)
library(tidyverse)
bac <- read.table("bacteria.txt",header = T,row.names = 1)

counts<-rowSums(bac>0)
bac <-bac[counts>=12,]#OTU
bac <-bac/29224
bac<-t(bac)
write.csv(bac, "Village-bacteria.csv")
calNetwork <- function(data = data,filter = FALSE,
                       n = 3,method = "spearman",
                       cutoff_cor = 0.6, cutoff_p = 0.05){
  require(WGCNA)
  require("multtest")
  require(reshape2)
  require(tidyverse)
  
  cutoff_cor <- cutoff_cor
  cutoff_p <- cutoff_p
  if(filter){
    dat2 <- data[,colSums(data!= "0") > n]
  } else{
    dat2 <- data[,colSums(data!= "0") > round(nrow(data)/2,0)]
  }
  set.seed(123)
  dat_net <- corAndPvalue(dat2,method = method)#calculate correlation and significance
  dat_cor <- dat_net$cor#select correlation
  dat_p <- dat_net$p#select p values
  
  dat_cor[upper.tri(dat_cor,diag=TRUE)]=NA
  dat_p[upper.tri(dat_p,diag=TRUE)]=NA
  dat_cor_melt <- melt(dat_cor)
  dat_p_melt <- melt(dat_p)
  colnames(dat_cor_melt) <- c("from","to","correlation")
  colnames(dat_p_melt) <- c("from","to","p")
  
  dat_cor_melt %>%
    left_join(dat_p_melt, by = c("from","to")) %>%
    dplyr::filter(p!="NA") %>%
    arrange(p) -> dat_net
  #adjust p values
  procs <- c("ABH")#select benjamini
  p.adjust <- mt.rawp2adjp(dat_net$p,procs)
  
  dat_net2 <- data.frame(dat_net,p.adjust$adjp) %>%
    dplyr::filter(abs(correlation)> cutoff_cor,ABH < cutoff_p) %>%
    dplyr::select(from:correlation,ABH)
  
  colnames(dat_net2) <- c("from","to","correlation","adjust.p")
  return(dat_net2)
}
calNetwork(bac)
calNetwork(bac,cutoff_cor = 0.7,cutoff_p = 0.01)

net_edge <- calNetwork(bac)
c( as.character(net_edge$from), as.character(net_edge$to)) %>%
  as_tibble() %>%
  group_by(value) %>%
  summarize(n=n()) -> vertices
colnames(vertices) <- c("name", "n")
vertices
calNetModule <- function(data = data){
  require(tidyverse)
  require(igraph)
  set.seed(123)
  
  modules <- cluster_fast_greedy(data)
  modules2 <- list()
  modules3 <- data.frame()
  for (i in seq_along(modules)) {
    modules2[[i]] <- modules[[i]] %>%
      as_tibble() %>%
      mutate(name = value) %>%
      select(-value) %>%
      mutate(module = i)
    modules3 <- rbind(modules3,
                      modules2[[i]])
  }
  dat <- list(modules,modules3)
  return(dat)
}#this is a function
calNetParameters <- function(data = data){
  require(igraph)
  set.seed(123)
  node_number <- length(V(data))#Nodes
  network_edge_number <- length(E(data))#edge
  network_Average_density <- graph.density(data)#density
  network_transitivity <- transitivity(data, type="global")
  network_diameter <- diameter(data)
  network_Average_path.length<- average.path.length(data)
  network_connectivity <- edge_connectivity(data)
  dat <- data.frame(node_number,network_edge_number,
                    network_Average_density,
                    network_transitivity,
                    network_diameter,
                    network_Average_path.length,
                    network_connectivity)
  return(dat)
}#this is a function
bac_network_edges <- calNetwork(bac)
dim(bac_network_edges)#edges
##nodes
bac_nodes <-c( as.character(bac_network_edges$from),
               as.character(bac_network_edges$to)) %>%
  as_tibble() %>%
  group_by(value) %>%
  summarize(n=n())

colnames(bac_nodes)=c("Otu_id","degree")
dim(bac_nodes)#nodes
###module analysis
bac_graph <- graph_from_data_frame(bac_network_edges, vertices = bac_nodes, directed = FALSE )
bac_network_m <- calNetModule(bac_graph)
bac_network_m

write.csv(bac_network_m[[2]],"City-bacteria-Ä£¿é1.csv")

bac_nodes2 <- bac_nodes %>%
  left_join(bac_network_m[[2]], by = "name")
bac_nodes2
##topology parameters analysis
calNetParameters(bac_graph)
write.csv(calNetParameters(bac_graph),"Village-fungi-topology.csv")
write.csv(bac_network_edges,"Village-fungi-edges1.csv")
write.csv(bac_nodes,"Village-fungi-nodes1.csv")
library(tidyverse)
library(igraph)
mygraph1 <- graph_from_data_frame(bac_network_edges, vertices = bac_nodes, directed = FALSE )
adj2 <- data.frame(get.adjacency(mygraph1,sparse=FALSE))
write.csv(adj2,"Village-fungi (0,1).csv")
adj2_w<-data.frame(get.adjacency(mygraph1,sparse=FALSE,attr="correlation"))
write.csv(adj2_w,"Village-bacteria-correaltion.csv")
