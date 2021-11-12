#generate data by yuima
set.seed(123)
library(coda)
library(mvtnorm)
library(yuima)
d=50
nu=5
z1=paste("b",c(1:d),sep="")
z2=paste("x",c(1:d),sep="")
z3=paste(z1,"+",z2,sep="")
z22=z3[1]
for(i in 2:d){
  z22=paste(z22,",",z3[i],sep="")
}
fs<-function(k){
  z1=paste("b",c(1:d),sep="")
  z2=paste("x",c(1:d),sep="")
  z3=paste(z1,"+",z2,sep="")
  z22=z3[1]
  for(i in 2:d){
    z22=paste(z22,",",z3[i],sep="")
  }
  z22=paste("c(",z22,")",sep="")
  z4=paste("-(nu+d)*sum(",z22,"*","bm[,",k,"]",")",sep="")
  z5=paste("(nu+sum(",z22,"%*%","bm","%*%",z22,"))",sep="")
  zp=paste(z4,"/",z5,sep="")
  return(zp)
}
sol <-paste("x",c(1:d),sep="")
sol

b<-matrix("0",d,d)
diag(b)="2^0.5"

a<-c()
for(i in 1:d){
  a[i]=fs(i)
}
bb=rmvt(1,sigma=diag(10.0,d,d),df=3);
bb=sort(bb)
#bb
ll<-list()
for(i in 1:d){
  z=paste("b",i,sep="")
  ll[[z]]=bb[i]
}
#ll
true.parameters <- ll
unlist(true.parameters)


library(yuima)
nd<-length(sol)
zz=rWishart(1,50,diag(1,nd,nd))
#zz
#eigen(zz[,,1])$value
bat=zz[,,1]
#eigen(bat)$value
bm=solve(bat)
n <- 100
T=5
ysamp <- setSampling(Terminal = T, n = n)
ymodel <- setModel(drift = a, diffusion = b, solve.variable = sol)

#true.parameters <- list(mu1=2,mu2=1,mu3=0,mu4=1,mu5=2)
yuima <- setYuima(model = ymodel, sampling = ysamp)
yuima <- simulate(yuima,xinit=0,true.parameter = true.parameters)

plot(yuima@data)
true.parameters

x<-yuima@data
xx<-as.matrix(x@original.data)
dt=T/n
delxx<-xx[2:(n+1),]-xx[1:n,]

par(mfrow=c(2,1))
plot(xx[,1],type="l")
plot(xx[,25],type="l")

####tuning for reference measure
Rcpp::sourceCpp('sdecpp.cpp')
nt=2e5
rest<-fradp(nt,g=1.0,g2=0.8,xx,delxx,dt,bm)
sig=diag(1,d,d)
accept(rest[[2]][(nt-5e3):nt,1:2])
lt<-c()
lt[[1]]<-cov(rest[[2]][1e4:nt,])
lt[[2]]<-colMeans(rest[[2]][1e4:nt,])

sigk2<-lt[[1]] ##precondition /mu
vvk2<-lt[[2]]  ##precondition /Sigma


###########################################

N=1e6
sig=sigk2
vvk=vvk2
#random walk
res1=fr(N,g=0.8,xx,delxx,dt,bm,sig)
accept(res1[[2]][(N-5e3):N,1:2])
#pcn kernel
res2=fpcn(N,rho=0.9,xx,delxx,dt,ps=1,bm,sig,vvk)
accept(res2[[2]][(N-2e3):N,1:2])
plot(res2[[1]][(N-1e3):N],type="l")
#mpcn kernel
res3=fmpcn(N,rho=0.65,xx,delxx,dt,ps=1,bm,sig,vvk)
accept(res3[[2]][(N-1e3):N,1:2])
plot(res3[[2]][(N-1e3):N,1],type="l")
#guided mpcn
res4=fgmpcn(N,rho=0.65,xx,delxx,dt,ps=1,bm,sig,vvk)
accept(res4[[2]][(N-1e3):N,1:2])

#inifinity hmc
res5=fsplit_hmc(N,xx,delxx,dt,L=1,ep=0.4,0.4,ps=1,bm,sig,vvk)##
#accept(res91[[2]][(N-1e3):N,1:2])
#huggop kernel
res6=fhughop(N,xx,delxx,dt,L=1,1,bm,sig,vvk,0.3,5,2) ##
#weave kernel
res7=fwea(N,xx,delxx,dt,L=1,ep1=0.35,ep2=0.35,bm,sig,vvk) ##
#haar weave
res8=fhw(N,xx,delxx,dt,L=1,ep1=0.36,ep2=0.36,bm,sig,vvk) ##


source("ess-function.R")
out=kf(c(1,2,3,4,5,6,7,8))
out