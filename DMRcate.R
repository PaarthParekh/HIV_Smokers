setwd('C:/Users/part/Desktop')
library(Biobase)
BiocManager::install("DMRcate")
library(DMRcate)
BiocManager::install("lumi") #for beta2m 
library(lumi)
library(limma)

#data data and assigning data matrix
discover = read.csv('replc_final.csv')
disc_val = discover[c(15:nrow(discover)),]

#Taking first column for row names
rownames(disc_val)<-disc_val[,1]
disc_val<-disc_val[,-1]

#Convert the dataframe in numeric
disc_val[] <- lapply(disc_val, function(x) { if(is.factor(x)) as.numeric(as.character(x)) else x
})

#Convert beta values to M values
#disc_val_M <-beta2m(disc_val)

#annotate the Beta value matrix
smokingdata<-as.data.frame(discover[6,2:ncol(discover)])
smoker<-lapply(smokingdata, function(x) {if(x==1) 1 else 0})
smoker<-as.numeric(smoker)

Intercept<-lapply(smokingdata, function(x) {if(x==1) 1 else 1})
Intercept<-as.numeric(Intercept)
design<-cbind(Intercept, smoker)
colnames(design)<-c("(Intercept)", "Smoking")
design<-apply(design,2, FUN=as.numeric)
rownames(design)<-c(1:529)

discvalmatrix=data.matrix(disc_val) #needs to be converted

memory.limit(size=20000)
annotation=cpg.annotate(datatype = c("array"), discvalmatrix, what=c("Beta"), arraytype=c("450K"), analysis.type = c("differential"), design, contrasts = FALSE, cont.matrix = NULL, fdr = 0.1, coef=2)

#Find DMR
dmrcoutput<-dmrcate(annotation, lambda = 200, C=2)

results.ranges <- extractRanges(dmrcoutput, genome = "hg19")
write.csv(results.ranges,'replc_dmrs_fdr_0.1.csv')

exportcsv(annotation@ranges, "replc_anno_ranges_0.1.csv")
exportcsv(annotation@ranges@elementMetadata, "replc_anno_meta_0.1.csv")
