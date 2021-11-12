#include <RcppArmadillo.h>
#include <Rcpp/Benchmark/Timer.h>
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;
using namespace arma;

static double const log2pi = std::log(2.0 * M_PI);

const double nu=5.0;
const double gt=1.5;
const int bn=2e4;

// set seed
// [[Rcpp::export]]
void set_seed(unsigned int seed) {
  Rcpp::Environment base_env("package:base");
  Rcpp::Function set_seed_r = base_env["set.seed"];
  set_seed_r(seed);  
}

// [[Rcpp::export]]

int fnn(int m){
  int lik=0;int a;
  for(int i=0;i<300;i++){
    lik +=std::pow(i,2);a=i;
    if(m>-1&&m<=lik){break;}
  }
  return (a);
}


// [[Rcpp::export]]
arma::mat myfun(int d) {
  set_seed(123);
  arma::mat ma(d,d);
  for(int i=0;i<d;i++){
    for(int j=0;j<d;j++){
      ma(i,j)=10.0*std::pow(0.2,std::abs(i-j));
      //ma(i,j)=R::rnorm(0,1.0);
    }
  }
  return(ma);
}



// [[Rcpp::export]]
double dmvnrm(arma::rowvec x,  
              arma::rowvec mean,  
              arma::mat sigma){  
  int xdim = x.n_cols;
  arma::mat invsigma=arma::inv(sigma);
  arma::mat const rooti = arma::inv(trimatu(arma::chol(sigma)));
  double const rootisum = arma::sum(log(rooti.diag())), 
    constants = -(double)xdim/2.0 * log2pi, other_terms = rootisum + constants;
  double out=other_terms - 0.5 * arma::sum(x*invsigma*x.t());
  //out=exp(out);
  return(out) ;
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


//loglikelihood

// [[Rcpp::export]]
double loglik(arma::rowvec alpha,arma::mat xx,arma::mat delxx,double dt,arma::mat insig){
  double sg=10.0;//nuu=3.0;// sigma and freedom
  double prior=-0.5*arma::sum(alpha%alpha)/(sg);
  int nn=delxx.col(0).n_elem; //observation number
  int n=delxx.row(0).n_elem;  //parameeter dimension
  //double prior=-0.5*(nuu+n)*std::log(1+(arma::sum(alpha%alpha)/sg)/nuu);
  
  double loglik=0;
  double K=(n+nu);
  arma::rowvec pv(n),bi(n),vx(n);
  for(int i=0;i<nn;i++){
    pv=xx.row(i)+alpha;
    bi=-K*(pv*insig)/(nu+arma::sum(pv*insig*pv.t()));
    vx=delxx.row(i)-dt*bi;
    loglik=(-0.5/(2.0*dt)*arma::sum(vx%vx))+loglik;
  }
  return(loglik+prior);
}



// [[Rcpp::export]]
arma::rowvec dif(arma::rowvec alpha,arma::mat xx,arma::mat delxx,double dt,arma::mat insig,int n,int nn){
  
  double sg=10.0;//nuu=3.0;
  //nn=delxx.col(0).n_elem; //observation number
  //n=delxx.row(0).n_elem;  //parameeter dimension
  
  arma::mat A;
  
  double K=(n+nu),da,aa;
  arma::rowvec dif(n),diff(n),pv(n),vx(n),pa(n);
  //dif=-(nuu+n)*(alpha/sg)/(nuu+arma::sum(alpha%alpha)/sg);
  dif=-alpha/sg;
  for(int i=0;i<nn;i++){
    pv=xx.row(i)+alpha;//alpha.subvec(5,9);
    pa=(pv*insig);
    aa=(nu+arma::sum(pa%pv));
    vx=delxx.row(i)-(-K*dt/(aa))*pa;
    A=aa*insig-2*pa.t()*pa;
    dif=-0.5*K/(aa*aa)*(vx*A)+dif;
    //dif=-0.5*K*vx*B+dif;
  }
  
  return(dif);
}



/* rwm*/
// [[Rcpp::export]]
List fr(int N, double g,arma::mat xx,arma::mat delxx,double dt,arma::mat insig,arma::mat presig){
  List out(4);
  
  //double lower=0.01,upper=5.0;
  
  
  arma::rowvec beta_fix(3);beta_fix.ones();
  
  int n=delxx.n_cols;
  arma::mat C=arma::chol(presig);
  
  arma::mat res(N,n);res.zeros();
  
  arma::rowvec rd(n),pro(n),log_like(N),zz(n);
  res.row(0)=rd.randn()*0.1;
  zz=res.row(0);
  log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  
  clock_t start=std::clock();
  
  
  for(int i=1;i<N;i++){
    rd=rd.randn();
    pro=res.row(i-1)+2.38*rd/(std::pow(n,0.5))*C*g;
    // if(sum(pro<lower)+sum(pro>upper)==0){
    //   break;
    // }
    double a=loglik(pro,xx,delxx,dt,insig);
    double acc=a-log_like(i-1);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
      
    }
    if(i==bn){start=std::clock();}
    zz=zz+res.row(i);
  }
  zz=zz/(1.0*N);
  
  
  double t4 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  //arma::mat ress = arma::join_rows(res,resa);
  //arma::mat lik = arma::join_cols(log_like,log_likea);
  arma::mat ress=res.cols(0,3);out(0)=log_like;out(1)=res;out(2)=t4;out(3)=zz;
  return(out);
}





// adaptive random walk Metopolis kernel
// [[Rcpp::export]]
List fradp(int N, double g,double g2,arma::mat xx,arma::mat delxx,double dt,arma::mat insig){
  List rm(2),out(3);
  
  int n=delxx.n_cols; 
  arma::rowvec rd(n),lk(N),pro(n),av_new(n);
  arma::mat res(N,n);res.zeros();res.row(0)=rd.randn();
  arma::mat sig(n,n);sig=sig.eye();
  arma::mat cholsig=arma::chol(sig);
  
  int M=1e4;
  rm=fr(M,g,xx,delxx,dt,insig,sig);
  arma::mat bmmat=rm(1); 
  arma::rowvec lk2=rm(0);
  double acc=accept(bmmat);
  Rcout << "The accept rate" << acc << "\n";
  
  arma::mat covmat=arma::cov(bmmat);
  arma::rowvec av=arma::mean(bmmat,0);
  arma::mat bpm=covmat*2.38*2.38/std::pow(n,1);
  arma::mat snew(n,n);snew=arma::chol(bpm);
  res.rows(0,M-1)=bmmat;lk.subvec(0,M-1)=lk2;
  clock_t start=std::clock();
  
  for(int i=M;i<N;i++){
    arma::rowvec vv_pro(n);vv_pro.randn(); //proposal  random vector
    pro=res.row(i-1)+vv_pro.randn()*snew; //new proposal
    double lik_new=loglik(pro,xx,delxx,dt,insig);
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
      arma::mat bpm=covmat*2.38*2.38/n*g2;
      snew=arma::chol(bpm);
    }
  }
  
  double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  
  acc=accept(res.rows(M,N-1));
  Rcout << "The accept rate" << acc << "\n";
  
  uvec IDX = regspace<uvec>(0,10,n-1);
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=res;out(2)=t4/1000.0;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}



//pcn
// [[Rcpp::export]]
List fpcn(int N, double rho,arma::mat xx,arma::mat delxx,double dt,int ps,arma::mat insig,arma::mat presig,arma::rowvec vv){
  List out(4);
  
  double lower=0.01,upper=5.0;
  
  arma::rowvec beta_fix(3);beta_fix.ones();
  int n=delxx.n_cols;
  
  arma::mat C(n,n);C=arma::chol(presig);
  arma::mat invsig=arma::inv(presig);
  arma::mat res(N,n);res.zeros();
  res.row(0).randn();
  res.row(0)=vv*0.9;
  
  clock_t start=std::clock();
  arma::rowvec rd(n),pro(n),log_like(N),zz(n);
  log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  
  double kx=arma::sum(((res.row(0)-vv)*invsig)%(res.row(0)-vv));
  double a,acc,u,kn;
  for(int i=1;i<N;i++){
    rd=rd.randn()*C;
    pro=vv+std::pow(rho,0.5)*(res.row(i-1)-vv)+std::pow(1.0-rho,0.5)*rd;
    a=loglik(pro,xx,delxx,dt,insig);
    kn=0.5*arma::sum(((pro-vv)*invsig)%(pro-vv));
    acc=a-log_like(i-1)+0.5*kn-0.5*kx;
    u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
      kx=kn;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
  }
  
  clock_t end=std::clock();
  double t4 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  
  arma::mat ress=res.cols(0,3);out(0)=log_like;out(1)=res;out(2)=t4;
  return(out);
}



//mpcn
// [[Rcpp::export]]
List fmpcn(int N, double rho,arma::mat xx,arma::mat delxx,double dt,int ps,arma::mat insig,arma::mat presig,arma::rowvec vv){
  List out(4);
  double lower=0.01,upper=5.0;
  
  int n=delxx.n_cols;
  
  arma::mat C(n,n);C=arma::chol(presig);
  arma::mat invsig=arma::inv(presig);
  arma:: mat invsnew=arma::chol(arma::inv(presig)).t();  
  arma::mat res(N,n);res.zeros();
  res.row(0).randn();
  res.row(0)=res.row(0)*C;
  
  arma::rowvec rd(n),pro(n),log_like(N),zz(n);
  log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  
  clock_t start=std::clock();
  double kx=arma::sum(((res.row(0)-vv)*invsig)%(res.row(0)-vv));
  double a,acc,u,kn;
  for(int i=1;i<N;i++){
    double gg=R::rgamma(0.5*n,2.0/kx);
    rd=rd.randn();
    pro=vv+std::pow(rho,0.5)*(res.row(i-1)-vv)+std::pow(1.0-rho,0.5)*rd*std::pow(gg,-0.5)*C;
    
    kn=arma::sum(((pro-vv)*invsig)%(pro-vv));
    a=loglik(pro,xx,delxx,dt,insig);
    acc=a-log_like(i-1)+0.5*n*std::log(kn)-0.5*n*std::log(kx);
    u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
      kx=kn;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
  }
  clock_t end=std::clock();
  
  double t3 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  arma::mat ress=res.cols(0,3);out(0)=log_like;out(1)=res;out(2)=t3;
  return(out);
}


/* guided mpcn*/
// [[Rcpp::export]]
List fgmpcn(int N, double rho,arma::mat xx,arma::mat delxx,double dt,int ps,arma::mat insig,arma::mat presig,arma::rowvec vv){
  List out(4);
  
  double lower=0.01,upper=5.0;
  int n=delxx.n_cols;
  arma::rowvec beta_fix(3);beta_fix.ones();
  
  arma::mat C(n,n);C=arma::chol(presig);
  arma::mat invsig=arma::inv(presig);
  arma:: mat invsnew=arma::chol(arma::inv(presig)).t();  
  arma::mat res(N,n);res.zeros();
  arma::rowvec rd(n),pro(n),log_like(N),zz(n),apro(n);
  
  //res.row(0)=rd.randn()*C;
  res.row(0)=vv*0.8;
  
  log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  
  //double sd=std::pow(100,0.5),sdd=sd*sd;
  clock_t start=std::clock();
  arma::rowvec orgv=(res.row(0)-vv)*invsnew;
  double a,acc,u,gg,bb;
  int v=1;
  double aa=arma::norm(orgv,2);
  for(int i=1;i<N;i++){
    if(v==1){
      for(;;){
        gg=R::rgamma(0.5*n,2.0/(aa*aa));
        rd=rd.randn();
        apro=std::pow(rho,0.5)*orgv+std::pow(1.0-rho,0.5)*rd*std::pow(gg,-0.5);
        bb=arma::norm(apro,2);
        if(bb>aa){
          break;
        }
      }
    }else{
      for(;;){
        gg=R::rgamma(0.5*n,2.0/(aa*aa));
        rd=rd.randn();
        apro=std::pow(rho,0.5)*orgv+std::pow(1.0-rho,0.5)*rd*std::pow(gg,-0.5);
        bb=arma::norm(apro,2);
        if(bb<aa){
          break;
        }
      }
    }
    pro=vv+apro*C;
    double a=loglik(pro,xx,delxx,dt,insig);
    double acc=a-log_like(i-1)+n*std::log(bb)-n*std::log(aa);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
      orgv=apro;
      aa=bb;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
      v=-v;
    }
    zz=zz+res.row(i);
  }
  
  
  clock_t end=std::clock();
  
  arma::rowvec alpha_fix=zz/(double)N;
  //
  double t4 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  
  arma::mat ress=res.cols(0,3);out(0)=log_like;out(1)=res;out(2)=t4,out(3)=vv;
  return(out);
}




// [[Rcpp::export]]
List fsplit_hmc(int N,arma::mat xx,arma::mat delxx,double dt,int L,double ep,double ep2,int ps,arma::mat insig,
                arma::mat presig,arma::rowvec vv){
  
  List out(4);
  int n=delxx.n_cols;int nn=delxx.col(0).n_elem;
  
  arma::mat C(n,n);C=arma::chol(presig);
  arma::mat invsig=arma::inv(presig);
  arma:: mat invsnew=arma::chol(arma::inv(presig)).t(); 
  arma::mat res(N,n);res.zeros();
  
  arma::rowvec rd(n),log_like(N),pro(n),log_likef(N);
  
  res.row(0)=rd.randn()*C;
  
  log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  
  
  
  //rho=0.4;//////######
  clock_t start=std::clock();
  
  //log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  
  arma::rowvec gx(n),gy(n),zp(n),zpro(n),zm_new(n),zv_new(n);
  arma::mat zm(L+1,n),zv(L+1,n);
  arma::mat comp=res;
  comp.row(0)=res.row(0)-vv;
  
  for(int i=1;i<N;i++){
    zm.zeros();
    zv=zv.randn()*C;
    zm.row(0)=comp.row(i-1);
    //zv.row(1)=zv.row(0)-0.5*ep*dif(zm.row(0));
    for(int j=1;j<(L+1);j++){
      zm_new=std::cos(0.5*ep)*zm.row(j-1)+std::sin(ep*0.5)*zv.row(j-1);
      zv_new=-std::sin(0.5*ep)*zm.row(j-1)+std::cos(ep*0.5)*zv.row(j-1);
      
      zv_new=zv_new+ep2*(dif(zm_new+vv,xx,delxx,dt,insig,n,nn)+zm_new*invsig)*presig;
      
      zm.row(j)=std::cos(0.5*ep)*zm_new+std::sin(ep*0.5)*zv_new;
      zv.row(j)=-std::sin(0.5*ep)*zm_new+std::cos(ep*0.5)*zv_new;
    }
    
    pro=zm.row(L);
    double a=loglik(pro+vv,xx,delxx,dt,insig);
    double acc=a-log_like(i-1)-0.5*arma::as_scalar((zv.row(L))*invsig*(zv.row(L)).t())+
      0.5*arma::as_scalar((zv.row(0))*invsig*(zv.row(0)).t());
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro+vv;
      log_like(i)=a;
      comp.row(i)=pro;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
      comp.row(i)=comp.row(i-1);
    }
    if((i%100000==0)){       Rcout << "The iter of v : " << i << "\n";     }
  }
  double t4 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  
  arma::mat ress=res.cols(0,3);out(0)=log_like;out(1)=res;out(2)=t4;
  return(out);
}


//haar weave
// [[Rcpp::export]]
List hw(arma::mat xx,arma::mat delxx,double dt,int N, arma::mat sig,arma::rowvec vv,arma::mat insig,
          arma::rowvec firstrow,int L,double up, double low){
  List out(2);int n=delxx.n_cols;int nn=delxx.col(0).n_elem;
  
  arma::mat C(n,n);C=arma::chol(sig);
  //sig=C;
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat ivsig=arma::inv(sig);
  arma::mat res(N,n);res.row(0)=firstrow;
  arma::rowvec log_like(N),pro(n),pro_t(n),v(N);
  log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  double bb,bbn;
  arma::rowvec rw(n),rd(n),zvv(n),zmm(n),df(n),zf(n);
  arma:: rowvec orgv=(res.row(0)-vv);
  double aa=arma::norm(orgv*invsnew,2);
  double gg=0,dk,ak1,ak2,g,a,acc,u;
  //L=arma::randi( distr_param(1,L));
  L=L+1;
  arma::mat zm(L,n),zv(L,n);//zm.zeros();
  //ak1=up,ak2=low;
  for(int i=1;i<N;i++){
    gg=R::rgamma(0.5*n,2.0/(aa*aa));
    g=std::sqrt(1.0/gg);
    //rw=rw.randn();rd=rd.randn();
    zv.row(0)=rw.randn()*g*C;
    //arma::rowvec fv=past-vv;
    arma::rowvec uk(L);uk=uk.randu()*(up-low)+low;
    ak1=uk(0);
    zm.row(0)=std::cos(ak1)*orgv+std::sin(ak1)*zv.row(0);
    zv.row(0)=-std::sin(ak1)*orgv+std::cos(ak1)*zv.row(0);
    for(int j=1;j<L;j++){
      ak2=uk(j);
      // dk=arma::sum(zm.row(j-1)*ivsig*zm.row(j-1).t());
      // df=-(dif(zm.row(j-1)+vv,x,Y)+n*zm.row(j-1)*ivsig/dk);
      // zf=df/(arma::norm(df,2));                 
      // zvv=zv.row(j-1)-2.0*(arma::sum(zf%zv.row(j-1)))*zf*sig/arma::sum(zf*sig*zf.t());
      rd=zm.row(j-1)*ivsig;
      df=-(dif(zm.row(j-1)+vv,xx,delxx,dt,insig,n,nn)+n*rd/arma::dot(rd,zm.row(j-1)));//arma::sum(rd%zm.row(j-1)));
      zf=df*sig;                
      zvv=zv.row(j-1)-2.0*(arma::dot(df,zv.row(j-1)))*zf/arma::dot(zf,df);
      //zmm=zm.row(j-1);
      zm.row(j)=std::cos(ak2)*zm.row(j-1)+std::sin(ak2)*zvv;
      zv.row(j)=-std::sin(ak2)*zm.row(j-1)+std::cos(ak2)*zvv;
    }
    pro_t=zm.row(L-1);
    bb=arma::norm(pro_t*invsnew,2);
    pro=vv+pro_t;
    a=loglik(pro,xx,delxx,dt,insig);
    acc=a-log_like(i-1)+n*std::log(bb/aa);
    u=std::log(arma::as_scalar(arma::randu(1)));
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
List fhw(int N,arma::mat xx,arma::mat delxx,double dt,int L,double ep1,double ep2,
           arma::mat insig,arma::mat presig,arma::rowvec vvk){
  List out(5),rm(2);
  int n=delxx.n_cols;int nn=delxx.col(0).n_elem;  arma::mat sig(n,n);sig=presig;
  arma::mat chl=arma::chol(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=rd.randn();
  lk(0)=loglik(res.row(0),xx,delxx,dt,insig);
  int M=1;
  
  //Rcout << "The ep" << ep << "\n";
  
  clock_t start=std::clock();
  rm=hw(xx,delxx,dt,(N-M),sig,vv,insig,res.row(0),L,ep1,ep2);
  //rm=hug3(x,Y,(N-M),sig,vv,res.row(0),L,up,low);
  
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  
  uvec IDX = regspace<uvec>(0,10,n-1);
  double t4 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=res;out(2)=t4;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}


//weave-kerenl
// [[Rcpp::export]]
List wea(arma::mat xx,arma::mat delxx,double dt,int N, arma::mat sig,arma::rowvec vv,arma::mat insig,
              arma::rowvec firstrow,int L,double up, double low){
  List out(2);int n=delxx.n_cols;int nn=delxx.col(0).n_elem;
  
  arma::mat C(n,n);C=arma::chol(sig);
  //sig=C;
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat ivsig=arma::inv(sig);
  arma::mat res(N,n);res.row(0)=firstrow;
  arma::rowvec log_like(N),pro(n),pro_t(n),v(N);
  log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  double bb,bbn;
  arma::rowvec rw(n),rd(n),zvv(n),zmm(n),df(n),zf(n);
  arma:: rowvec orgv=(res.row(0)-vv);
  double aa=arma::norm(orgv*invsnew,2);
  double gg=0,dk,ak1,ak2,g,a,acc,u;
  //L=arma::randi( distr_param(1,L));
  L=L+1;
  arma::mat zm(L,n),zv(L,n);//zm.zeros();
  //ak1=up,ak2=low;
  for(int i=1;i<N;i++){
    //rw=rw.randn();rd=rd.randn();
    zv.row(0)=rw.randn()*C;
    //arma::rowvec fv=past-vv;
    arma::rowvec uk(L);uk=uk.randu()*(up-low)+low;
    ak1=uk(0);
    zm.row(0)=std::cos(ak1)*orgv+std::sin(ak1)*zv.row(0);
    zv.row(0)=-std::sin(ak1)*orgv+std::cos(ak1)*zv.row(0);
    for(int j=1;j<L;j++){
      ak2=uk(j);
      // dk=arma::sum(zm.row(j-1)*ivsig*zm.row(j-1).t());
      // df=-(dif(zm.row(j-1)+vv,x,Y)+n*zm.row(j-1)*ivsig/dk);
      // zf=df/(arma::norm(df,2));                 
      // zvv=zv.row(j-1)-2.0*(arma::sum(zf%zv.row(j-1)))*zf*sig/arma::sum(zf*sig*zf.t());
      rd=zm.row(j-1)*ivsig;
      //df=-(dif(zm.row(j-1)+vv,xx,delxx,dt,insig,n,nn)+n*rd/arma::dot(rd,zm.row(j-1)));//arma::sum(rd%zm.row(j-1)));
      df=-(dif(zm.row(j-1)+vv,xx,delxx,dt,insig,n,nn)+rd);
      zf=df*sig;                
      zvv=zv.row(j-1)-2.0*(arma::dot(df,zv.row(j-1)))*zf/arma::dot(zf,df);
      //zmm=zm.row(j-1);
      zm.row(j)=std::cos(ak2)*zm.row(j-1)+std::sin(ak2)*zvv;
      zv.row(j)=-std::sin(ak2)*zm.row(j-1)+std::cos(ak2)*zvv;
    }
    pro_t=zm.row(L-1);
    bb=arma::norm(pro_t*invsnew,2);
    pro=vv+pro_t;
    a=loglik(pro,xx,delxx,dt,insig);
    acc=a-log_like(i-1)+0.5*(bb*bb-aa*aa);
    u=std::log(arma::as_scalar(arma::randu(1)));
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
List fwea(int N,arma::mat xx,arma::mat delxx,double dt,int L,double ep1,double ep2,
               arma::mat insig,arma::mat presig,arma::rowvec vvk){
  List out(5),rm(2);
  int n=delxx.n_cols;int nn=delxx.col(0).n_elem;  arma::mat sig(n,n);sig=presig;
  arma::mat chl=arma::chol(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=vv*0.9;
  lk(0)=loglik(res.row(0),xx,delxx,dt,insig);
  int M=1;
  
  //Rcout << "The ep" << ep << "\n";
  
  clock_t start=std::clock();
  rm=wea(xx,delxx,dt,(N-M),sig,vv,insig,res.row(0),L,ep1,ep2);
  //rm=hug3(x,Y,(N-M),sig,vv,res.row(0),L,up,low);
  
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  
  uvec IDX = regspace<uvec>(0,10,n-1);
  double t4 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  double acc=accept(bm2);
  Rcout << "The accept rate" << acc << "\n";
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=res;out(2)=t4;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}

//hug hop kerenl
// [[Rcpp::export]]
List hughop(arma::mat xx,arma::mat delxx,double dt,int N, arma::mat sig,arma::rowvec vv,arma::mat insig,
            arma::rowvec firstrow,int L,int L2,double ep,double lamda,double m){
  
  
  
  int n=delxx.n_cols;int nn=delxx.col(0).n_elem;
  List out(4);
  arma::mat C(n,n),I(n,n);C=arma::chol(sig);I.eye();
  //sig=C;
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat ivsig=arma::inv(sig);
  arma::mat res(N,n);res.row(0)=firstrow;
  arma::rowvec log_like(N),pro(n),pro_t(n),v(N);
  log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
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
      df=-dif(zmm+vv,xx,delxx,dt,insig,n,nn);
      zf=df/(arma::norm(df,2));                 
      zv.row(j)=zv.row(j-1)-2.0*(arma::sum(zf%zv.row(j-1)))*zf*sig/arma::sum(zf*sig*zf.t());
      //zmm=zm.row(j-1);
      zm.row(j)=zv.row(j)*ep+zmm;
    }
    
    pro_t=zm.row(L);
    //bb=arma::sum(zm.row(L)*ivsig*zm.row(L-1).t());
    pro=vv+pro_t;
    a=loglik(pro,xx,delxx,dt,insig);
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
      arma::rowvec dx=dif(orgv+vv,xx,delxx,dt,insig,n,nn);
      //arma::mat B=(m*I+(lamda-m)*gx.t()*gx)/nx;
      arma::rowvec bx=dx*sig;double bxn=arma::dot(bx,dx);
      //arma::mat BB=(1.0/(m*m)*sig+(1.0/(lamda*lamda)-1.0/(m*m))*gx.t()*gx)*(nx*nx);
      arma::mat BB=((m*m)*sig+(lamda*lamda-m*m)/bxn*bx.t()*bx)/(bxn);
      arma::mat BP=arma::chol(BB);
      rd=rd.randn();
      pro_t=orgv+rd*BP;pro=pro_t+vv;
      arma::rowvec dy=dif(pro,xx,delxx,dt,insig,n,nn);
      arma::rowvec by=dy*sig;double byn=arma::dot(by,dy);
      arma::mat BBY=((m*m)*sig+(lamda*lamda-m*m)/byn*by.t()*by)/(byn);
      arma::mat invBBY=arma::inv(BBY);
      a=loglik(pro,xx,delxx,dt,insig);
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
List fhughop(int N,arma::mat xx,arma::mat delxx,double dt,int L,int L2,
             arma::mat insig,arma::mat presig,arma::rowvec vvk,double ep,double lamda,double m){
  List out(5),rm(2);
  int n=delxx.n_cols;int nn=delxx.col(0).n_elem;  arma::mat sig(n,n);sig=presig;
  
  arma::mat chl=arma::chol(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec rd(n),lk(N),pro(n),pro_t(n),v(n),vv(n);vv=vvk;
  arma::mat res(N,n);res.row(0)=rd.randn();
  lk(0)=loglik(res.row(0),xx,delxx,dt,insig);
  int M=1;
  
  //Rcout << "The ep" << ep << "\n";
  
  clock_t start=std::clock();
  //test=hughop(xx,delxx,dt,(N-M), sig,vvk,bm,rnorm(d),1,1,0.3,12,3)
  rm=hughop(xx,delxx,dt,(N-M),sig,vv,insig,res.row(0),L,L2,ep,lamda,m);
  
  arma::mat bm2=rm(1); arma::rowvec lk2=rm(0);
  res.rows(M,N-1)=bm2;lk.subvec(M,N-1)=lk2;
  
  uvec IDX = regspace<uvec>(0,10,n-1);
  //double t4 = (std::clock() - start )*1000/CLOCKS_PER_SEC;
  double t4 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  double acc=accept(bm2),acc1=rm(2),acc2=rm(3);
  Rcout << "The accept rate" << acc << "\n";
  Rcout << "The accept rate hug " << acc1 << "\n";
  Rcout << "The accept rate hop" << acc2 << "\n";
  arma::mat ress=res.cols(IDX);
  out(0)=lk;out(1)=res;out(2)=t4;//out(3)=sec(res.rows(1e4,N-1));
  return(out);
}


