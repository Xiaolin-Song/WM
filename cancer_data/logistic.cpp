#include <RcppArmadillo.h>
#include <Rcpp/Benchmark/Timer.h>
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends("mcmcse")]]

#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;
using namespace arma;

static double const log2pi = std::log(2.0 * M_PI);
const double sdd=10.0; //prior
const double nu=10.0; //prior


// [[Rcpp::export]]

int fnn(int m){
  int lik=0;int a;
  for(int i=0;i<300;i++){
    lik +=std::pow(i,2);a=i;
    if(m>-1&&m<=lik){break;}
  }
  return (a);
}

/* accept rate*/
// [[Rcpp::export]]
double accept(arma::mat x){
  int nn=x.n_rows;
  int s=0;
  for(int i=1;i<nn;i++){
    //arma::umat ss=(x.row(i)==x.row(i-1));
    if(x(i,0)==x(i-1,0)){
      s=s;
    }else{
      s=s+1;
    }
  }
  double ac=s/(double)nn;
  return(ac);
}





// log-likelihood
// [[Rcpp::export]]
double loglik(arma::rowvec x,arma::mat xx,arma::rowvec yy){
  double df=1.0;int dim=x.size();
  double aa=-0.5*(df+dim)*std::log(1+arma::sum(x%x)/df);
  //double aa=arma::sum(-arma::log(1+x.subvec(1,nx-1)%x.subvec(1,nx-1)/(xc*nuu)))-std::log(1+x(0)*x(0)/(100.0*nuu));
  //double aa=-0.5*arma::sum((x)*x.t())/(sdd);
  int n=yy.n_elem;
  double K,KK;
  for(int i=0;i<n;i++){
    K=arma::sum(-x%xx.row(i));
    KK=(1.0+std::exp(K));
    if(yy(i)==1){
      aa=-std::log(KK)+aa;
    }else{
      aa=std::log(1-1.0/KK)+aa;
    }
    
  }
  return(aa);
}


//gradient for log-likelihood
// [[Rcpp::export]]
arma::rowvec dif(arma::rowvec x,arma::mat xx,arma::rowvec yy){
  //int n=xx.n_rows;
  double df=1.0;
  int dimx=x.size();
  //arma::rowvec dif=-x/sdd;
  arma::rowvec dif=-(df+dimx)/(df+arma::sum(x%x))*x,difp(dimx);
  arma::vec y=yy.t()-1.0/(1.0+arma::exp(-xx*x.t()));
  //arma::vec y=yy.t()-sig;
  for(int j=0;j<dimx;j++){
    difp(j)=arma::sum(y%xx.col(j));
  }
  return(dif+difp);
}


//gradient for log-likelihood
// [[Rcpp::export]]
arma::rowvec dif2(arma::rowvec x,arma::mat xx,arma::rowvec yy,arma::mat invsig){
  int n=xx.n_rows;
  double df=1.0;int nx=x.size();
  arma::rowvec dif(nx);
  dif=-(df+nx)/(df+arma::sum(x%x))*x+nx*x*invsig/arma::sum((x*invsig)%x);
  //dif.subvec(1,nx-1)=-2*x.subvec(1,nx-1)/(xc*nuu+x.subvec(1,nx-1)%x.subvec(1,nx-1));
  //dif(0)=-2*x(0)/(100*nuu+x(0)*x(0));
  //dif=-x/sdd+n*x*invsig/arma::sum((x*invsig)%x);
  for(int i=0;i<n;i++){
    double sigm=1.0/(1.0+std::exp(arma::sum(-x%xx.row(i))));
    dif=(yy(i)-sigm)*xx.row(i)+dif;
  }
  return(dif);
}






// random walk Metopolis kernel
// [[Rcpp::export]]
List rwm(arma::mat x,arma::rowvec Y,int N, arma::mat cholsig, double g,
         arma::rowvec firstrow){
  List out(2);
  int n=x.n_cols;arma::mat res(N,n);res.row(0)=firstrow;
  arma::rowvec rd(n),log_like(N),pro(n);
  res.row(0)=firstrow;
  log_like(0)=loglik(res.row(0),x,Y);
  for(int i=1;i<N;i++){
    rd=rd.randn()*cholsig;
    pro=res.row(i-1)+2.38*rd/(std::pow(n,0.5))*g;
    double a=loglik(pro,x,Y);
    double acc=a-log_like(i-1);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
  }
  out(0)=log_like;out(1)=res;
  return(out);
}


// adaptive random walk Metopolis kernel
// [[Rcpp::export]]
List fr(arma::mat x,arma::rowvec Y,int N,double g,int pt){
  List rm(2),out(4);
  int n=x.n_cols;
  arma::rowvec rd(n),lk(N),pro(n),av_new(n);
  arma::mat res(N,n);res.zeros();res.row(0)=rd.randn();
  arma::mat sig(n,n);sig=sig.eye();
  arma::mat cholsig=arma::chol(sig);
  
  int M=1e4;
  rm=rwm(x,Y,M,cholsig,g,res.row(0));
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  
  arma::mat covmat=arma::cov(bm2);
  arma::rowvec av=arma::mean(bm2,0);
  arma::mat bpm=covmat*2.38*2.38/std::pow(n,1);
  arma::mat snew(n,n);snew=arma::chol(bpm);
  res.rows(0,M-1)=bm2;lk.subvec(0,M-1)=lk2;
  clock_t start=std::clock();
  
  for(int i=M;i<N;i++){
    arma::rowvec vv_pro(n);vv_pro.randn(); //proposal  random vector
    pro=res.row(i-1)+vv_pro.randn()*snew; //new proposal
    double lik_new=loglik(pro,x,Y);
    double u=arma::as_scalar(arma::randu(1));
    double alpha=lik_new-lk(i-1);
    if(alpha>std::log(u)){
      res.row(i)=pro;
      lk(i)=lik_new;
    }else{
      res.row(i)=res.row(i-1);
      lk(i)=lk(i-1);
    }
    av_new=(i*av+res.row(i))/(i+1);
    arma::mat md1=(i*av.t()*av-(i+1)*(av_new.t()*av_new)+res.row(i).t()*res.row(i));
    covmat=((i-1)*covmat+md1)/i;
    av=av_new;
    
    if(fnn(i-M+1)-fnn(i-M)>0){
      arma::mat bpm=covmat*2.38*2.38/n*0.8;
      snew=arma::chol(bpm);
    }
  }
  
  
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  
  acc=accept(res.rows(M,N-1));
  Rcout << "The accept rate" << acc << "\n";
  
  uvec IDX = regspace<uvec>(0,10,n-1);
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=res;out(2)=t4/1000.0;out(3)=covmat;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}

// random walk Metopolis kernel
// [[Rcpp::export]]
List frr(arma::mat x,arma::rowvec Y,int N,double g,arma::mat sigk){
  List rm(2),out(3);
  int n=x.n_cols;
  arma::rowvec rd(n),lk(N);
  arma::mat res(N,n);res.zeros();res.row(0)=rd.randn();
  arma::mat sig(n,n);sig=sigk;
  arma::mat cholsig=arma::chol(sig);
  
  int M=1;
  clock_t start=std::clock();
  rm=rwm(x,Y,(N-M),cholsig,g,res.row(0));
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  
  uvec IDX = regspace<uvec>(0,10,n-1);
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=res;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}



// guided Metropolis-Hastings kernel
// [[Rcpp::export]]
List gmpcn(arma::mat x,arma::rowvec Y,int N, arma::mat sig,arma::rowvec vv, double rho,
           arma::rowvec firstrow){
  List out(2);int n=x.n_cols;
  arma::mat C(n,n);C=arma::chol(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat res(N,n);res.row(0)=firstrow;
  arma::rowvec rd(n),log_like(N),pro(n),pro_t(n),v(N);
  res.row(0)=firstrow;
  log_like(0)=loglik(res.row(0),x,Y);
  double bb,bbn,r1,r2;
  
  r1=std::pow(rho,0.5);r2=std::pow(1.0-rho,0.5);
  arma:: rowvec orgv=(res.row(0)-vv)*invsnew;
  double aa=arma::norm(orgv,2);
  double gg=0;v(0)=1;
  for(int i=1;i<N;i++){
    if(v(i-1)==1){
      for(;;){
        rd=rd.randn();
        gg=R::rgamma(0.5*n,2.0/(aa*aa));
        pro_t=r1*orgv+r2*rd*std::pow(gg,-0.5);
        bb=arma::norm(pro_t,2);
        if(bb<aa){
          break;
        }
      }
    }else{
      for(;;){
        rd=rd.randn();
        gg=R::rgamma(0.5*n,2.0/(aa*aa));
        pro_t=r1*orgv+r2*rd*std::pow(gg,-0.5);
        bb=arma::norm(pro_t,2);
        if(bb>aa){
          break;
        }
      }
    }
    pro=vv+pro_t*C;
    double a=loglik(pro,x,Y);
    double acc=a-log_like(i-1)+n*std::log(bb/aa);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
      v(i)=v(i-1);
      orgv=pro_t;
      aa=bb;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
      v(i)=-1*v(i-1);
    }
    //Rcout << "The accept rate" << i << "\n";
  }
  out(0)=log_like;out(1)=res;
  return(out);
}



// [[Rcpp::export]]
List fgmpcn(arma::mat x,arma::rowvec Y,int N,double rho,arma::mat sigk,arma::rowvec vvk){
  List out(5),rm(2);
  int n=x.n_cols;
  arma::mat sig(n,n);sig=sigk;
  
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=rd.randn();
  lk(0)=loglik(res.row(0),x,Y);
  int M=1;
  Rcout << "The rho" << rho << "\n";
  
  clock_t start=std::clock();
  
  rm=gmpcn(x,Y,(N-M),sig,vv,rho,res.row(0));
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  //res=bm2;lk=lk2;
  uvec IDX = regspace<uvec>(0,10,n-1);
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=res;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}


//mpcn kernel
// [[Rcpp::export]]
List mpcn(arma::mat x,arma::rowvec Y,int N, arma::mat sig,arma::rowvec vv, double rho,
          arma::rowvec firstrow){
  List out(2);int n=x.n_cols;
  arma::mat C(n,n);C=arma::chol(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat res(N,n);res.row(0)=firstrow;
  arma::rowvec rd(n),log_like(N),pro(n),pro_t(n),v(N);
  log_like(0)=loglik(res.row(0),x,Y);
  double bb,bbn,r1,r2;
  
  r1=std::pow(rho,0.5);r2=std::pow(1.0-rho,0.5);
  arma:: rowvec orgv=(res.row(0)-vv)*invsnew;
  double aa=arma::norm(orgv,2);
  double gg=0;v(0)=1;
  for(int i=1;i<N;i++){
    rd=rd.randn();
    gg=R::rgamma(0.5*n,2.0/(aa*aa));
    pro_t=r1*orgv+r2*rd*std::pow(gg,-0.5);
    bb=arma::norm(pro_t,2);
    pro=vv+pro_t*C;
    double a=loglik(pro,x,Y);
    double acc=a-log_like(i-1)+n*std::log(bb/aa);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
      orgv=pro_t;
      aa=bb;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
    //Rcout << "The accept rate" << i << "\n";
  }
  out(0)=log_like;out(1)=res;
  return(out);
}

// [[Rcpp::export]]
List fmpcn(arma::mat x,arma::rowvec Y,int N,double rho,arma::mat sigk,arma::rowvec vvk){
  List out(5),rm(2);
  int n=x.n_cols;
  arma::mat sig(n,n);sig=sigk;
  
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=rd.randn();
  lk(0)=loglik(res.row(0),x,Y);
  int M=1;
  Rcout << "The rho" << rho << "\n";
  
  clock_t start=std::clock();
  
  rm=mpcn(x,Y,(N-M),sig,vv,rho,res.row(0));
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  //res=bm2;lk=lk2;
  uvec IDX = regspace<uvec>(0,10,n-1);
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=res;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}


//pcn kernel 
// [[Rcpp::export]]
List pcn(arma::mat x,arma::rowvec Y,int N,arma::mat sig,arma::rowvec vv,double rho,
         arma::rowvec firstrow){
  List out(2);int n=x.n_cols;
  arma::mat C(n,n);C=arma::chol(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat invsig=arma::inv(sig);
  arma::mat res(N,n);res.row(0)=firstrow;
  arma::rowvec rd(n),log_like(N),pro(n),pro_t(n),v(N);
  log_like(0)=loglik(res.row(0),x,Y);
  double bb,bbn,r1,r2;
  
  r1=std::pow(rho,0.5);r2=std::pow(1.0-rho,0.5);
  arma:: rowvec orgv=(res.row(0)-vv)*invsnew;
  double aa=arma::sum(orgv%orgv);
  double gg=0;v(0)=1;
  for(int i=1;i<N;i++){
    rd=rd.randn()*C;
    pro=vv+r1*(res.row(i-1)-vv)+r2*rd;
    double a=loglik(pro,x,Y);
    double bb=arma::sum((pro-vv)*invsig*(pro-vv).t());
    double acc=a-log_like(i-1)+0.5*bb-0.5*aa;
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
      aa=bb;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
    //Rcout << "The accept rate" << i << "\n";
  }
  out(0)=log_like;out(1)=res;
  return(out);
}

// [[Rcpp::export]]
List fpcn(arma::mat x,arma::rowvec Y,int N,double rho,arma::mat sigk,arma::rowvec vvk){
  List out(5),rm(2);
  int n=x.n_cols;
  arma::mat sig(n,n);sig=sigk;
  
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=rd.randn();
  lk(0)=loglik(res.row(0),x,Y);
  int M=1;
  clock_t start=std::clock();
  
  rm=pcn(x,Y,(N-M),sig,vv,rho,res.row(0));
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  //res=bm2;lk=lk2;
  uvec IDX = regspace<uvec>(0,10,n-1);
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=res;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}




// split hmc kernel
// [[Rcpp::export]]
List split_hmc(arma::mat x,arma::rowvec Y,int N,double ep,double ep2,int L,arma::mat sigk,arma::rowvec vvk,
               arma::rowvec firstrow){
  
  int n=x.n_cols;
  arma::mat sig=sigk;
  List out(2);
  
  arma::mat clp(n,n);clp=arma::chol(sig);
  arma::mat invsig=arma::inv(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat res(N,n);res.zeros();
  res.row(0)=firstrow;
  arma::rowvec rd(n),log_like(N),pro(n),vv(n);vv=vvk;
  arma::rowvec zp(n),zpro(n),zm_new(n),zv_new(n);
  log_like(0)=loglik(res.row(0),x,Y);
  arma::mat zm(L+1,n),zv(L+1,n);
  arma::rowvec past=res.row(0)-vv;
  arma::rowvec diff;
  double ak1=std::cos(ep*0.5),ak2=std::sin(ep*0.5);
  for(int i=1;i<N;i++){
    rd=rd.randn();
    zv.row(0)=rd*clp;
    zm.row(0)=past;
    //zv.row(1)=zv.row(0)-0.5*ep*dif(zm.row(0));
    for(int j=1;j<(L+1);j++){
      
      zm_new=ak1*zm.row(j-1)+ak2*zv.row(j-1);
      zv_new=-ak2*zm.row(j-1)+ak1*zv.row(j-1)+ep2*(dif(zm_new+vv,x,Y)*sig+zm_new);
      
      //diff=dif(zm_new+vv,x,Y);
      //zv_new=zv_new+ep2*(dif(zm_new+vv,x,Y)*sig+zm_new);
      zm.row(j)=ak1*zm_new+ak2*zv_new;
      zv.row(j)=-ak2*zm_new+ak1*zv_new;
    }
    
    pro=zm.row(L);
    arma::rowvec prot=pro+vv;
    double a=loglik(prot,x,Y);
    //arma::rowvec rf=zv.row(L)*invsnew;
    double acc=a-log_like(i-1)-0.5*arma::sum((zv.row(L)*invsig)%zv.row(L))+0.5*arma::sum(rd%rd);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=prot;
      log_like(i)=a;
      past=pro;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
  }
  out(0)=log_like;out(1)=res;
  return(out);
}

// [[Rcpp::export]]
List fsplit_hmc(arma::mat x,arma::rowvec Y,int N,double ep,double ep2,
                int L,arma::mat sigk,arma::rowvec vvk,arma::rowvec
                  first){
  List out(5),rm(2);
  int n=x.n_cols;
  arma::mat sig(n,n);sig=sigk;
  arma::mat chl=arma::chol(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=first;
  lk(0)=loglik(res.row(0),x,Y);
  int M=1;
  
  Rcout << "The ep" << ep << "\n";
  
  clock_t start=std::clock();
  
  rm=split_hmc(x,Y,(N-M),ep,ep2,L,sig,vv,res.row(0));
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  
  uvec IDX = regspace<uvec>(0,10,n-1);
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=res;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}

//haar-weave
// [[Rcpp::export]]
List hw(arma::mat x,arma::rowvec Y,int N, arma::mat sig,arma::rowvec vv,
         arma::rowvec firstrow,int L,double up, double low){
  List out(2);int n=x.n_cols;
  arma::mat C(n,n);C=arma::chol(sig);
  //sig=C;
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat res(N,n);res.row(0)=firstrow;
  arma::rowvec log_like(N),pro(n),pro_t(n),v(N);
  log_like(0)=loglik(res.row(0),x,Y);
  double bb,bbn;
  arma::rowvec rw(n),rd(n),zvv(n),zmm(n),df(n),zf(n);
  arma:: rowvec orgv=(res.row(0)-vv)*invsnew;
  double aa=arma::norm(orgv,2);
  double gg=0,dk,ak1,ak2;
  //L=arma::randi( distr_param(1,L));
  L=L+1;
  arma::mat zm(L,n),zv(L,n);
  for(int i=1;i<N;i++){
    gg=R::rgamma(0.5*n,2.0/(aa*aa));
    zv.row(0)=std::sqrt(1.0/gg)*rw.randn();
    
    //arma::rowvec uk(L);uk=uk.randu()*(up-low)+low;
    ak1=up;
    //ak1=uk(0);
    zm.row(0)=std::cos(ak1)*orgv+std::sin(ak1)*zv.row(0);
    zv.row(0)=-std::sin(ak1)*orgv+std::cos(ak1)*zv.row(0);
    if(L>1){
      for(int j=1;j<L;j++){
        //uk=uk.randu()*(up-low)+low;
        ak2=low;//ak2=uk(1);
        dk=arma::sum(zm.row(j-1)%zm.row(j-1));
        df=-(dif(zm.row(j-1)*C+vv,x,Y)*C.t()+n*zm.row(j-1)/dk);
        zf=df/(arma::norm(df,2));                 
        zvv=zv.row(j-1)-2.0*(arma::dot(zf,zv.row(j-1)))*zf;
        //zmm=zm.row(j-1);
        zm.row(j)=std::cos(ak2)*zm.row(j-1)+std::sin(ak2)*zvv;
        zv.row(j)=-std::sin(ak2)*zm.row(j-1)+std::cos(ak2)*zvv;
        
      }
    }
    
    pro_t=zm.row(L-1);
    bb=arma::norm(pro_t,2);
    pro=vv+pro_t*C;
    double a=loglik(pro,x,Y);
    double acc=a-log_like(i-1)+n*std::log(bb/aa);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
      orgv=pro_t;
      aa=bb;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
    //Rcout << "The accept rate" << i << "\n";
  }
  out(0)=log_like;out(1)=res;
  return(out);
}

// //hugkernel
// // [[Rcpp::export]]
// List hug3(arma::mat x,arma::rowvec Y,int N, arma::mat sig,arma::rowvec vv,
//          arma::rowvec firstrow,int L,double up, double low){
//   List out(2);int n=x.n_cols;
//   arma::mat C(n,n);C=arma::chol(sig);
//   //sig=C;
//   arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
//   arma::mat ivsig=arma::inv(sig);
//   arma::mat res(N,n);res.row(0)=firstrow;
//   arma::rowvec log_like(N),pro(n),pro_t(n),v(N);
//   log_like(0)=loglik(res.row(0),x,Y);
//   double bb,bbn;
//   arma::rowvec rw(n),rd(n),zvv(n),zmm(n),df(n),zf(n);
//   arma:: rowvec orgv=(res.row(0)-vv);
//   double aa=arma::norm(orgv*invsnew,2);
//   double gg=0,dk,ak1,ak2,g,a,acc,u;
//   //L=arma::randi( distr_param(1,L));
//   L=L+1;
//   arma::mat zm(L,n),zv(L,n);//zm.zeros();
//   for(int i=1;i<N;i++){
//     gg=R::rgamma(0.5*n,2.0/(aa*aa));
//     g=std::sqrt(1.0/gg);
//     //rw=rw.randn();rd=rd.randn();
//     zv.row(0)=rw.randn()*g*C;
//     //arma::rowvec fv=past-vv;
//     arma::rowvec uk(L);uk=uk.randu()*(up-low)+low;
//     ak1=uk(0);
//     zm.row(0)=std::cos(ak1)*orgv+std::sin(ak1)*zv.row(0);
//     zv.row(0)=-std::sin(ak1)*orgv+std::cos(ak1)*zv.row(0);
//       for(int j=1;j<L;j++){
//         uk=uk.randu()*(up-low)+low;
//         ak1=uk(0),ak2=uk(1);
//         //ak2=uk(j);
//         // dk=arma::sum(zm.row(j-1)*ivsig*zm.row(j-1).t());
//         // df=-(dif(zm.row(j-1)+vv,x,Y)+n*zm.row(j-1)*ivsig/dk);
//         // zf=df/(arma::norm(df,2));                 
//         // zvv=zv.row(j-1)-2.0*(arma::sum(zf%zv.row(j-1)))*zf*sig/arma::sum(zf*sig*zf.t());
//         rd=zm.row(j-1)*ivsig;
//         df=-(dif(zm.row(j-1)+vv,x,Y)+n*rd/arma::dot(rd,zm.row(j-1)));//arma::sum(rd%zm.row(j-1)));
//         zf=df*sig;                
//         zvv=zv.row(j-1)-2.0*(arma::dot(df,zv.row(j-1)))*zf/arma::dot(zf,df);
//         //zmm=zm.row(j-1);
//         zm.row(j)=std::cos(ak2)*zm.row(j-1)+std::sin(ak2)*zvv;
//         zv.row(j)=-std::sin(ak2)*zm.row(j-1)+std::cos(ak2)*zvv;
//     }
//     pro_t=zm.row(L-1);
//     bb=arma::norm(pro_t*invsnew,2);
//     pro=vv+pro_t;
//     a=loglik(pro,x,Y);
//     acc=a-log_like(i-1)+n*std::log(bb/aa);
//     u=std::log(arma::as_scalar(arma::randu(1)));
//     if(acc>u){
//       res.row(i)=pro;
//       log_like(i)=a;
//       orgv=pro_t;
//       aa=bb;
//     }else{
//       res.row(i)=res.row(i-1);
//       log_like(i)=log_like(i-1);
//     }
//     //Rcout << "The accept rate" << i << "\n";
//   }
//   out(0)=log_like;out(1)=res;
//   return(out);
// }

// [[Rcpp::export]]
List fhw(arma::mat x,arma::rowvec Y,int N,int L,arma::mat sigk,arma::rowvec vvk,double up,double low){
  List out(5),rm(2);
  int n=x.n_cols;
  arma::mat sig(n,n);sig=sigk;
  arma::mat chl=arma::chol(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=rd.randn();
  lk(0)=loglik(res.row(0),x,Y);
  int M=1;
  
  //Rcout << "The ep" << ep << "\n";
  
  clock_t start=std::clock();
  
  rm=hw(x,Y,(N-M),sig,vv,res.row(0),L,up,low);
  //rm=hug3(x,Y,(N-M),sig,vv,res.row(0),L,up,low);
  
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  
  uvec IDX = regspace<uvec>(0,10,n-1);
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=res;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}

// weave kerenl
// [[Rcpp::export]]
List wea(arma::mat x,arma::rowvec Y,int N, arma::mat sig,arma::rowvec vv,
               arma::rowvec firstrow,int L,double up, double low){
  List out(2);int n=x.n_cols;
  arma::mat C(n,n);C=arma::chol(sig);
  //sig=C;
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat res(N,n);res.row(0)=firstrow;
  arma::rowvec log_like(N),pro(n),pro_t(n),v(N);
  log_like(0)=loglik(res.row(0),x,Y);
  double bb,bbn;
  arma::rowvec rw(n),rd(n),zvv(n),zmm(n),df(n),zf(n);
  arma:: rowvec orgv=(res.row(0)-vv)*invsnew;
  double aa=arma::norm(orgv,2);
  double gg=0,dk,ak1,ak2;
  //L=arma::randi( distr_param(1,L));
  L=L+1;
  arma::mat zm(L,n),zv(L,n);//zm.zeros();
  for(int i=1;i<N;i++){
    zv.row(0)=rw.randn();
    //arma::rowvec uk(L);uk=uk.randu()*(up-low)+low;ak1=uk(0);
    ak1=up;
    zm.row(0)=std::cos(ak1)*orgv+std::sin(ak1)*zv.row(0);
    zv.row(0)=-std::sin(ak1)*orgv+std::cos(ak1)*zv.row(0);
    if(L>1){
      for(int j=1;j<L;j++){
        //uk=uk.randu()*(up-low)+low;ak1=uk(0);
        ak2=low;//ak2=uk(1);//ak2=uk(j);
        dk=arma::dot(zm.row(j-1),zm.row(j-1));
        df=-(dif(zm.row(j-1)*C+vv,x,Y)*C.t()+n*zm.row(j-1)/dk);
        zf=df/(arma::norm(df,2));                 
        zvv=zv.row(j-1)-2.0*(arma::dot(zf,zv.row(j-1)))*zf;
        //zmm=zm.row(j-1);
        zm.row(j)=std::cos(ak2)*zm.row(j-1)+std::sin(ak2)*zvv;
        zv.row(j)=-std::sin(ak2)*zm.row(j-1)+std::cos(ak2)*zvv;
      }
    }
    
    pro_t=zm.row(L-1);
    bb=arma::norm(pro_t,2);
    pro=vv+pro_t*C;
    double a=loglik(pro,x,Y);
    double acc=a-log_like(i-1)+0.5*(bb*bb-aa*aa);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
      orgv=pro_t;
      aa=bb;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
    //Rcout << "The accept rate" << i << "\n";
  }
  out(0)=log_like;out(1)=res;
  return(out);
}

// [[Rcpp::export]]
List fwea(arma::mat x,arma::rowvec Y,int N,int L,arma::mat sigk,arma::rowvec vvk,double up,double low){
  List out(5),rm(2);
  int n=x.n_cols;
  arma::mat sig(n,n);sig=sigk;
  arma::mat chl=arma::chol(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=rd.randn();
  lk(0)=loglik(res.row(0),x,Y);
  int M=1;
  
  //Rcout << "The ep" << ep << "\n";
  
  clock_t start=std::clock();
  
  //rm=hug(x,Y,(N-M),sig,vv,res.row(0),L,up,low);
  rm=wea(x,Y,(N-M),sig,vv,res.row(0),L,up,low);
  
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  
  uvec IDX = regspace<uvec>(0,10,n-1);
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=res;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}

//hug&hop kerenl
// [[Rcpp::export]]
List hughop(arma::mat x,arma::rowvec Y,int N, arma::mat sig,arma::rowvec vv,
            arma::rowvec firstrow,int L,int L2,double ep,double lamda,double m){
  List out(4);int n=x.n_cols;
  arma::mat C(n,n),I(n,n);C=arma::chol(sig);I.eye();
  //sig=C;
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat ivsig=arma::inv(sig);
  arma::mat res(N,n);res.row(0)=firstrow;
  arma::rowvec log_like(N),pro(n),pro_t(n),v(N);
  log_like(0)=loglik(res.row(0),x,Y);
  arma::rowvec rw(n),rd(n),zvv(n),zmm(n),df(n),zf(n);
  arma:: rowvec orgv=(res.row(0)-vv);
  double aa,bb;
  double gg=0,dk,ak1,ak2,g,a,acc,u;double kn;
  //L=arma::randi( distr_param(1,L));
  aa=arma::sum(orgv*ivsig*orgv.t());double nga=0,ngp=0,nhug=0,nhop=0;
  for(int i=1;i<N;i++){
    kn=arma::as_scalar(arma::randu(1));
    double likold=log_like(i-1);
    arma::mat zm(L+1,n),zv(L+1,n);//zm.zeros();
    rw=rw.randn();
    zv.row(0)=rw*C;
    zm.row(0)=orgv;
    for(int j=1;j<(L+1);j++){
      //ak2=uk(j);
      zmm=zv.row(j-1)*ep+zm.row(j-1);
      df=-dif(zmm+vv,x,Y);
      zf=df/(arma::norm(df,2));                 
      zv.row(j)=zv.row(j-1)-2.0*(arma::sum(zf%zv.row(j-1)))*zf*sig/arma::sum(zf*sig*zf.t());
      //zmm=zm.row(j-1);
      zm.row(j)=zv.row(j)*ep+zmm;
    }
    
    pro_t=zm.row(L);
    //bb=arma::sum(zm.row(L)*ivsig*zm.row(L-1).t());
    pro=vv+pro_t;
    a=loglik(pro,x,Y);
    acc=a-log_like(i-1)-0.5*arma::dot(zv.row(L)*ivsig,zv.row(L))+0.5*arma::dot(rw,rw);;
    u=std::log(arma::as_scalar(arma::randu(1)));
    //kn=-1;
    nhug=nhug+1.0;
    if(acc>u){nga=nga+1.0;};
    if(acc>u){
      orgv=pro_t;
      likold=a;
    }
    // hop kernel
    for(int j=0;j<(L2);j++){
      arma::rowvec dx=dif(orgv+vv,x,Y);
      //arma::mat B=(m*I+(lamda-m)*gx.t()*gx)/nx;
      arma::rowvec bx=dx*sig;double bxn=arma::dot(bx,dx);
      //arma::mat BB=(1.0/(m*m)*sig+(1.0/(lamda*lamda)-1.0/(m*m))*gx.t()*gx)*(nx*nx);
      arma::mat BB=((m*m)*sig+(lamda*lamda-m*m)/bxn*bx.t()*bx)/(bxn);
      arma::mat BP=arma::chol(BB);
      rd=rd.randn();
      pro_t=orgv+rd*BP;pro=pro_t+vv;
      arma::rowvec dy=dif(pro,x,Y);
      arma::rowvec by=dy*sig;double byn=arma::dot(by,dy);
      arma::mat BBY=((m*m)*sig+(lamda*lamda-m*m)/byn*by.t()*by)/(byn);
      arma::mat invBBY=arma::inv(BBY);
      a=loglik(pro,x,Y);
      arma::rowvec dc=orgv-pro_t;
      acc=a-likold-0.5*arma::sum(dc*invBBY*dc.t())+0.5*arma::sum(rd*rd.t())+0.5*n*std::log(byn/bxn);
      u=std::log(arma::as_scalar(arma::randu(1)));
      //kn=1;
      nhop=nhop+1.0;
      if(acc>u){
        ngp=ngp+1.0;
        orgv=pro_t;
        likold=a;
      };
    }
    res.row(i)=orgv+vv;
    log_like(i)=likold;
  }
  
  
  //Rcout << "The accept rate" << i << "\n";
  out(0)=log_like;out(1)=res;out(2)=nga/(1.0*nhug);out(3)=ngp/(1.0*nhop);
  return(out);
}

// [[Rcpp::export]]
List fhughop(arma::mat x,arma::rowvec Y,int N,int L,int L2,arma::mat sigk,arma::rowvec vvk,double ep,double lamda,double m){
  List out(5),rm(2);
  int n=x.n_cols;
  arma::mat sig(n,n);sig=sigk;
  arma::mat chl=arma::chol(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=rd.randn();
  lk(0)=loglik(res.row(0),x,Y);
  int M=1;
  
  //Rcout << "The ep" << ep << "\n";
  
  clock_t start=std::clock();
  
  //rm=hug(x,Y,(N-M),sig,vv,res.row(0),L,up,low);
  rm=hughop(x,Y,(N-M),sig,vv,res.row(0),L,L2,ep,lamda,m);
  
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  
  uvec IDX = regspace<uvec>(0,10,n-1);
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  double acc=accept(bm2),acc1=rm(2),acc2=rm(3);
  Rcout << "The accept rate" << acc << "\n";
  Rcout << "The accept rate hug " << acc1 << "\n";
  Rcout << "The accept rate hop" << acc2 << "\n";
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=res;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}





