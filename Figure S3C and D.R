rm(list=ls())
library("VennDiagram","UpSetR")
#Read the relative abundance table of genus level
Data = read.table("otus.profile.txt", header=T, row.names= 1, sep="\t", comment.char="")

design = read.table("design.txt", header=T, row.names= 1, sep="\t", comment.char="")

index = rownames(design) %in% colnames(Data)
design = design[index,]
Data = Data[,rownames(design)]

# Add group information to the data table
Data_t = t(Data)
Data_t2 = merge(design, Data_t, by="row.names")

Data_t2 = Data_t2[,c(-1,-3)]

# Take the mean by group
# Define the mean function of each group
Data_mean = aggregate(Data_t2[,-1], by=Data_t2[1], FUN=mean)
# Mean for each group
Data4Pic = as.data.frame(do.call(rbind, Data_mean)[-1,])
colnames(Data4Pic) = Data_mean$group
Data4Pic=as.data.frame(Data4Pic)

# Replace the value >0 in the data table with 1. If the value >0, OTU or data at all levels appear in the group. Replace the value with 1 for data preparation for visualization%
Data4Pic[Data4Pic>0]=1

write.table(Data4Pic,"d3.data4venn.txt", sep="\t", quote=F)
Data4Pic = read.table("d3.data4venn.txt", header=T, row.names=1)
pdf(file="p1.GenusVenn.pdf", width=4, height=3, pointsize=8)
p1 <- venn.diagram(
  x=list(A=row.names(Data4Pic[Data4Pic$A==1,]),
         B=row.names(Data4Pic[Data4Pic$B==1,]),
         C=row.names(Data4Pic[Data4Pic$C==1,]),
         D=row.names(Data4Pic[Data4Pic$D==1,])),
  filename = NULL, lwd = 3, alpha = 0.6,
  label.col = "white", cex = 1.5,
  fill = c("dodgerblue", "goldenrod1", "darkorange1", "seagreen3"),
  cat.col = c("dodgerblue", "goldenrod1", "darkorange1", "seagreen3"),
  fontfamily = "serif", fontface = "bold",
  cat.fontfamily = "serif",cat.fontface = "bold",
  margin = 0.05)
grid.draw(p1)
dev.off()
png(file="p1.GenusVenn.png", width=4, height=3, res=300, units="in")
grid.draw(p1)
dev.off()
row.names(Data4Pic) <- 1:nrow(Data4Pic)
library(UpSetR)
pdf(file="p2.GenusUpset.pdf", width=4, height=3, pointsize=8)
(p2 <-upset(Data4Pic, sets = colnames(Data4Pic),order.by = "freq"))
dev.off()
png(file="p2.GenusUpset.png", width=4, height=3, res=300, units="in")
p2

dev.off()
