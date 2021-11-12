library(loo)
library(coda)
library(mcmcse)
library(dirmcmc)
set.seed(124)
sd=rcauchy(1,scale = 10)
phi=0.4
h<-c()
yy<-c()
h[1]=rnorm(1)*sd/(1-phi^2)^0.5
yy[1]<-rnorm(1)*exp(h[1]/2)
T=100
for(i in 2:T){
  h[i]=phi*h[i-1]+rnorm(1)*sd
  yy[i]<-rnorm(1)*exp(h[i]/2)
}

n=length(yy)+2
X<-diag(1,n,n)
Y<-yy

####tuning for reference measure
Rcpp::sourceCpp('sv.cpp')
nt=2e5
t1<-Sys.time()
rest<-fr(X,Y,nt,0.4,pt=0)
t2<-Sys.time()
t2-t1
#plot(rest[[1]][1e5:nt],type="l")
lt<-c()
lt[[1]]<-rest[[4]]
lt[[2]]<-colMeans(rest[[2]][2e4:nt,])
# # save(lt,file="lt.Rdata")
# # load("lt.Rdata")
sigk2<-lt[[1]] ##precondition /mu
vvk2<-lt[[2]]  ##precondition /Sigma


####################################
sigk=sigk2
vvk=vvk2
N=1e6
t1<-Sys.time()
#random walk
res_1<-frr(X,Y,N,1.0,sigk)
#pcn
res_2<-fpcn(X,Y,N,0.96,sigk,vvk,vvk*0.8)
#mpcn
res_3<-fmpcn(X,Y,N,0.96,sigk,vvk)
#gmpcn
res_4<-fgmpcn(X,Y,N,0.95,sigk,vvk)
#infinity hcm
res_5<-fsplit_hmc(X,Y,N,0.5,0.5,1,sigk,vvk,vvk2*0.9)
#hughop
res_6<-fhughop(X,Y,N,L=1,1,sigk2,vvk2,0.3,12,3)
#weave
res_7<-fwea(X,Y,N,L=1,sigk2,vvk2,0.3,0.2,vvk2*0.9)
#haar weave
res_8<-fhw(X,Y,N,L=1,sigk,vvk,0.3,0.2,vvk2*0.8)


source("ess-function.R")
out=kf(c(1,2,3,4,5,6,7,8))
out
