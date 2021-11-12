library(mlbench)
library(mcmcse)
#data=read.csv('C:/Users/song/Desktop/onarnew/github-sonar/sonar', header=FALSE)
data=read.csv('sonar', header=FALSE)

dim(data)
data<-data[,1:61]
dim(data)
XX<-data[,1:60]
YY<-data[,61]
YY<-as.numeric(YY)
YY<-YY-1
YY
dd=length(YY)
X<-as.matrix(XX)
X<-scale(X)*0.5
Y<-YY


library(mlbench)
set.seed(1237)

##tuning for reference measure
Rcpp::sourceCpp('logistic.cpp')
nt=1e5
rest<-fr(X,Y,nt,0.4,pt=0)
lt<-c()
lt[[1]]<-cov(rest[[2]][1e4:nt,])
lt[[2]]<-colMeans(rest[[2]][1e4:nt,])
# save(lt,file="lt.Rdata")
# load("lt.Rdata")
sigk2<-lt[[1]] ##precondition /mu
vvk2<-lt[[2]]  ##precondition /Sigma

#Rcpp::sourceCpp('logistic.cpp')



N=1e6
t1<-Sys.time()
#random walk
res_1<-frr(X,Y,N,1.0,sigk2)
#pcn
res_2<-fpcn(X,Y,N,0.4,sigk2,vvk2)
#mpcn
res_3<-fmpcn(X,Y,N,0.5,sigk2,vvk2)
#gmpcn
res_4<-fgmpcn(X,Y,N,0.5,sigk2,vvk2)
#inifinite hmc
res_5<-fsplit_hmc(X,Y,N,0.9,0.9,1,sigk2,vvk2,vvk2*0.5)
#hug&gop
res_6<-fhughop(X,Y,N,L=1,1,sigk2,vvk2,0.3,12,3)
#weave
res_7<-fwea(X,Y,N,L=1,sigk2,vvk2,0.55,0.55,vvk2*1.1)
#haar-weave
res_8<-fhw(X,Y,N,L=1,sigk2,vvk2,0.55,0.55)
#res_10<-fmix_split(X,Y,N,L=1,ep=0.6,0.25,sigk2,vvk2,vvk2*0.5)

source("ess-function.R")
output=kf2(c(1,2,3,4,5,6,7,8))
output


