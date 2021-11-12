library(loo)
library(coda)
cancer<-read.table("wdbc.data",sep = ",",
                   dec = ".",
                   quote = "'",header = FALSE)

dim(cancer)
dm=cancer[,3:32]
dn=scale(dm)*0.5


Y<-as.numeric(cancer[,2])

for(i in 1:dim(cancer)[1]){
  if(Y[i]==2){
    Y[i]=0
  }
}
n=dim(cancer)[1]
X<-as.matrix(dn)
X=cbind(rep(1,n),X)
Y<-as.matrix(Y)
dd=dim(X)[2]


Rcpp::sourceCpp('logistic.cpp')

##tuning for the reference measure with adaptive MCMC
nt=1e5
rest<-fr(X,Y,nt,0.4,pt=0)
lt<-c()
lt[[1]]<-cov(rest[[2]][1e4:nt,])
lt[[2]]<-colMeans(rest[[2]][1e4:nt,])
sigk2<-lt[[1]] ##precondition /mu
vvk2<-lt[[2]]  ##precondition /Sigma

N=1e6
t1<-Sys.time()
#random walk metropolis
res_1<-frr(X,Y,N,1.0,sigk2)
#pcn
res_2<-fpcn(X,Y,N,0.3,sigk2,vvk2)
#mpcn
res_3<-fmpcn(X,Y,N,0.3,sigk2,vvk2)
#guided mpcn
res_4<-fgmpcn(X,Y,N,0.3,sigk2,vvk2)
#inifinite hmc
res_5<-fsplit_hmc(X,Y,N,0.85,0.85,1,sigk2,vvk2,vvk2*0.5)
#hug&gop
res_6<-fhughop(X,Y,N,L=1,1,sigk2,vvk2,0.3,12,3)
#weave
res_7<-fwea(X,Y,N,L=1,sigk2,vvk2,0.58,0.58,vvk2*0.95)
#haar-weave
res_8<-fhw(X,Y,N,L=1,sigk2,vvk2,0.6,0.6)

source("ess-function.R")
output=kf2(c(1,2,3,4,5,6,7,8))
output

