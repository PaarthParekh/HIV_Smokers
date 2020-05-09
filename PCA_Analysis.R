setwd('C:/Users/paart/Desktop')

#data data and assigning data matrix
discover = read.csv('replc_top1000_nonnull_2.csv', row.names = 1)
discover = read.csv('top 86.csv', row.names = 1)
tdiscover = as.data.frame(t(discover))
tdisc=as.data.frame(tdiscover[,14:ncol(tdiscover)])

#convert to numeric
tdisc[] <- lapply(tdisc, function(x) { as.double(as.character(x))})


#Plotting PCA
library(ggfortify)
tdiscover = as.data.frame(t(discover))
colnames(tdiscover)[1]="separator" #choose required column number for having colour codes 
autoplot(prcomp(tdisc, center=TRUE, scale. = TRUE), data = tdiscover, colour = "separator")
autoplot(prcomp(tdisc, center=FALSE, scale. = FALSE), data = tdiscover, colour = "separator")
