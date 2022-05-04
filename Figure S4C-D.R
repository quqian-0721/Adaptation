
library(randomForest)
otu <- read.delim('Fungal-RF.txt', row.names = 1)

n<-length(names(otu)) 
rate=1  
for(i in 1:(n-1)){
  set.seed(2)
  rf_train<-randomForest(FPC1~.,data=otu,mtry=i,importance = TRUE,ntree=500)
  rate[i]<-mean(rf_train)  
  print(rf_train)    
}
rf_train   
plot(rate)
set.seed(2)
rf_train<-randomForest(FPC1~.,data=otu,mtry=10,importance = TRUE,ntree=1000)
plot(rf_train) 

set.seed(2)
otu_forest <- randomForest(FPC1~., data = otu, importance = TRUE, proximity=TRUE,mtry=10,ntree = 500)
otu_forest

importance_otu.scale <- data.frame(importance(otu_forest, scale = TRUE), check.names = FALSE)
importance_otu.scale

importance_otu.scale <- importance_otu.scale[order(importance_otu.scale$'%IncMSE', decreasing = TRUE), ]

library(ggplot2)

importance_otu.scale$OTU_name <- rownames(importance_otu.scale)
importance_otu.scale$OTU_name <- factor(importance_otu.scale$OTU_name, levels = importance_otu.scale$OTU_name)

p <- ggplot(importance_otu.scale, aes(OTU_name, `%IncMSE`)) +
  geom_col(width = 0.5, fill = '#FFC068', color = NA) +
  labs(title = NULL, x = NULL, y = 'Increase in MSE (%)', fill = NULL) +
  theme(panel.grid = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = 'black')) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(expand = c(0, 0), limit = c(0, 7))

p

library(rfPermute)
set.seed(2)
otu_rfP <- rfPermute(FPC1~., data = otu, importance = TRUE, mtry=10,ntree = 500, nrep = 1000, num.cores = 1)
otu_rfP

importance_otu.scale <- data.frame(importance(otu_rfP, scale = TRUE), check.names = FALSE)
importance_otu.scale

importance_otu.scale.pval <- (otu_rfP$pval)[ , , 2]
importance_otu.scale.pval

importance_otu.scale <- importance_otu.scale[order(importance_otu.scale$'%IncMSE', decreasing = TRUE), ]
library(ggplot2)

importance_otu.scale$OTU_name <- rownames(importance_otu.scale)
importance_otu.scale$OTU_name <- factor(importance_otu.scale$OTU_name, levels = importance_otu.scale$OTU_name)

p <- ggplot() +
  geom_col(data = importance_otu.scale, aes(x = OTU_name, y = `%IncMSE`), width = 0.5, fill = '#FFC068', color = NA) +
  labs(title = NULL, x = NULL, y = 'Increase in MSE (%)', fill = NULL) +
  theme(panel.grid = element_blank(), panel.background = element_blank(), axis.line = element_line(colour = 'black')) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_y_continuous(expand = c(0, 0), limit = c(0, 8))

p

for (OTU in rownames(importance_otu.scale)) {
  importance_otu.scale[OTU,'%IncMSE.pval'] <- importance_otu.scale.pval[OTU,'%IncMSE']
  if (importance_otu.scale[OTU,'%IncMSE.pval'] >= 0.05) importance_otu.scale[OTU,'%IncMSE.sig'] <- ''
  else if (importance_otu.scale[OTU,'%IncMSE.pval'] >= 0.01 & importance_otu.scale[OTU,'%IncMSE.pval'] < 0.05) importance_otu.scale[OTU,'%IncMSE.sig'] <- '*'
  else if (importance_otu.scale[OTU,'%IncMSE.pval'] >= 0.001 & importance_otu.scale[OTU,'%IncMSE.pval'] < 0.01) importance_otu.scale[OTU,'%IncMSE.sig'] <- '**'
  else if (importance_otu.scale[OTU,'%IncMSE.pval'] < 0.001) importance_otu.scale[OTU,'%IncMSE.sig'] <- '***'
}

p <- p +
  geom_text(data = importance_otu.scale, aes(x = OTU_name, y = `%IncMSE`, label = `%IncMSE.sig`), nudge_y = 1)

p

library(A3)

set.seed(345)
otu_forest.pval <- a3(BPC1~., data = otu, model.fn = randomForest, p.acc = 0.001, model.args = list(importance = TRUE, mtry=8,ntree = 100))
otu_forest.pval
