rm(list=ls())
library(lavaan)
library(vegan)
library(tidyverse)
library(Hmisc)

env<-as.matrix(dat)
co_linear <- varclus(env, similarity="spear") # spearman is the default
co_linear
plot(co_linear)#
setwd("C:/Users/曲乾/Desktop/R")
dat<-read.csv("SEM-FSA.csv",row.names = 1,header=T)
dat<-data.frame(scale(dat,center = F))

modela <- '
# regressions
Net~ace+SSC+Dead.Live
Dead.Live~Ni+NH4.N+DOC
ace~Dead.Live+NH4.N+Chla+SSC+Ni
SSC~Ni+NH4.N+DOC
Chla~Ni+NH4.N+Dead.Live+DOC
'
#fit <- sem(model1, data = dat)
abioticCompFit1 <- sem(modela, missing="ml",data=dat)
fitMeasures(abioticCompFit1,c("chisq","df","pvalue","gfi","cfi","rmr","srmr","rmsea"))
summary(abioticCompFit1, rsquare=T, standardized=T)
#summary(abioticCompFit1, fit.measures=TRUE)
library(semPlot)
semPaths(abioticCompFit1)
semPaths(abioticCompFit1, what = "std", layout = "tree", fade=F, nCharNodes = 0, intercepts = F, residuals = F, thresholds = F) 
semPaths(abioticCompFit,layout = "spring")
