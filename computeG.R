#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

list.of.packages <- c("ggplot2", "psych", "lavaan","Hmisc","corrplot","semPlot","colorRamps", "GPArotation")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(ggplot2)
library(psych)
library(lavaan)
library(Hmisc)
library(corrplot)
library(semPlot)
library(colorRamps)
library(GPArotation)

sink("output_computeG.txt")

# Helpers functions
# compute Comparative Fit Index for a factor analysis 
CFI <-function(x){
  return((1-((x$STATISTIC-x$dof))/(x$null.chisq-x$null.dof)))
}
# compute Comparative Fit Index for a bifactor analysis 
CFI_biv <-function(x){
  return((1-((x$stats$STATISTIC-x$stats$dof))/(x$stats$null.chisq-x$stats$null.dof)))
}
# compute implied matrix for a factor analysis
impliedMatrix<-function(x){
  if (dim(x$loadings)[2]==1) {
    imp      <- x$loadings %*% t(x$loadings) 
  } else {
    imp      <- x$loadings %*% x$Phi %*% t(x$loadings) 
  }
  diag(imp)<- diag(imp) + x$uniquenesses
  return(imp)
}
# compute implied matrix for a bifactor analysis
impliedMatrix_biv<-function(x){
  Gloadings     <- x$schmid$sl[,1]
  Floadings     <- x$schmid$sl[,2:(ncol(x$schmid$sl)-3)]
  uniquenesses  <- x$schmid$sl[,ncol(x$schmid$sl)-1]
  imp           <- Gloadings %*% t(Gloadings) + Floadings %*% t(Floadings)
  diag(imp)     <- diag(imp) + uniquenesses
  return(imp)
}

cogScores = c('PicVocab_Unadj',             # Vocabulary, Language, Crystallized, Global
             'ReadEng_Unadj',               # Reading, Language, Crystallized, Global
             'PicSeq_Unadj',                # Episodic memory, Fluid, Global
             'Flanker_Unadj',               # Executive, Fluid, Global
             'CardSort_Unadj',              # Executive, Fluid, Global
             'ProcSpeed_Unadj',             # Speed, Executive, Fluid, Global
             'PMAT24_A_CR',                 # non-verbal reasoning: Number of Correct Responses, Median Reaction Time for Correct Responses 
             'VSPLOT_TC',                   # Spatial ability: Total Number Correct, Median Reaction Time Divided by Expected Number of Clicks for Correct 
             'IWRD_TOT',                    # Verbal memory
             'ListSort_Unadj'               # Working memory, Executive, Fluid, Global
            )
alpha = 1e-3
unrestricted = read.csv(args[1])
mysubs = read.table(args[2])
cogdf      = unrestricted[cogScores]
cogdf      = cogdf[unrestricted$Subject %in% mysubs$V1,]
# standardize scores
cogdf = scale(cogdf)

out = fa.parallel(cogdf,plot=F)#error.bars=T,se.bars=F,
faValues = out$fa.values
faSim    = out$fa.sim
faSimR   = out$fa.simr

fm     <- "mle"       # use maximum likelihood estimator
rotate <- "oblimin"   # use oblimin factor rotation

fitInds <- matrix(nrow = 2, ncol = 9)
rownames(fitInds) <- c('s1','b4')
colnames(fitInds) <- c('CFI','RMSEA','SRMR','BIC','om_h','om_s1','om_s2','om_s3','om_s4')

# observed covariance matrices
obs       <-  cov(cogdf)
lobs      <-  obs[!lower.tri(obs)]

#SINGLE FACTOR
model = 1
f1     <- fa(cogdf,nfactors=1)
imp    <-  impliedMatrix(f1)
limp   <-  imp[!lower.tri(imp)]
fitInds[model,1] <-  CFI(f1)
fitInds[model,2] <-  f1$RMSEA[1]
fitInds[model,3] <-  sqrt(mean((limp - lobs)^2))
fitInds[model,4] <-  f1$BIC
write.table(f1$scores,'f1Scores.csv',col.names = FALSE,row.names = FALSE)


# BI-FACTOR MODEL
model = 2
b4      <- omega(cogdf,nfactors=4,fm=fm,key=NULL,flip=FALSE,
                 digits=3,title="Omega",sl=TRUE,labels=NULL, plot=FALSE,
                 n.obs=NA,rotate=rotate,Phi = NULL,option="equal",covar=FALSE)
imp     <-  impliedMatrix_biv(b4)
limp    <-  imp[!lower.tri(imp)]
fitInds[model,1] <-  CFI_biv(b4)
fitInds[model,2] <-  b4$schmid$RMSEA[1]
fitInds[model,3] <-  sqrt(mean((limp - lobs)^2))
fitInds[model,4] <-  b4$stats$BIC
fitInds[model,5] <-  b4$omega_h
fitInds[model,6:9] <-  b4$omega.group[-1,3]

cat('\n## fitInds\n')
print(fitInds,digits=3)
cat("\n## b4\n")
print(b4)

pdf("b4.pdf") 
diagram(b4,digits=3,cut=.2)
dev.off()
# export scores
b4Scores    <- factor.scores(cogdf,b4$schmid$sl[,1:5])$scores
write.table(b4Scores,'b4Scores_EFA.csv',row.names = FALSE)

#Factor labels: 
#g   = General factor; 
#spd = Processing Speed; 
#cry = Crystallized Ability; 
#vis = Visuospatial Ability; 
#mem = Memory

#biB enforces loadings of 1 for factors defined by only two observed variables
biB <- '
    #g-factor
    g   =~ CardSort_Unadj + Flanker_Unadj + ProcSpeed_Unadj + PicVocab_Unadj + ReadEng_Unadj + PMAT24_A_CR + VSPLOT_TC + IWRD_TOT + PicSeq_Unadj
    #Domain factors
    spd =~ CardSort_Unadj + Flanker_Unadj + ProcSpeed_Unadj
    cry =~ 1*PicVocab_Unadj + 1*ReadEng_Unadj
    vis =~ 1*PMAT24_A_CR    + 1*VSPLOT_TC    
    mem =~ 1*IWRD_TOT       + 1*PicSeq_Unadj
    #Domain factors are not correlated with g
    g ~~ 0*spd
    g ~~ 0*cry
    g ~~ 0*vis
    g ~~ 0*mem
    #Domain factors are not correlated with one another
    spd ~~ 0*cry
    spd ~~ 0*vis
    spd ~~ 0*mem
    cry ~~ 0*vis
    cry ~~ 0*mem
    vis ~~ 0*mem
'
mod_biB    <- cfa(biB, data=cogdf,estimator='ML')
cat("\n## mod_biB\n")
print(mod_biB)

pdf('mod_biB.pdf')
semPaths(mod_biB, "model", "std", bifactor = "g", layout = "tree2", exoCov = FALSE, nCharNodes=0,sizeMan = 9,sizeMan2 = 4,XKCD=TRUE)#, residuals = FALSE
dev.off()
cat("\n## fitMeasures\n")
print(fitMeasures(mod_biB,c("cfi","tli","rmsea","srmr","aic","bic","chisq","df")))
# factor scores
biScores    = lavPredict(mod_biB)
write.table(biScores,'biScores_CFA.csv',row.names = FALSE)
sink(NULL)