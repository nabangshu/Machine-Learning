#include<iostream>
#include<cmath>
#include<cstring>
#include<cstdlib>
#include <fstream>
using namespace std;
int main(int argc,char* argv[]) {
	
	string line;
	string address=argv[1];
	int i,a=atoi(argv[2]),b=atoi(argv[3]),len,temp,count=0,aprior,bprior;
	double likelihood, fac1=1.0, fac2=1.0,k;
  	ifstream myfile (address.c_str());
  	if (myfile.is_open())
  	{
    	while ( getline (myfile,line) )
    	{
    		count++;
    		aprior=a; bprior=b;
    		a=0; b=0;
    		fac1=1.0; fac2=1.0;
    		len=line.length();
      		char *cstr = new char [len+1];
  			strcpy (cstr, line.c_str());
  			for(i=0;i<len;++i)
  				if (cstr[i]=='1')
  					a++;
  			b=i-a;
  			temp=(i-a>a)?i-a:a;
  			cout.precision(18);
  			for(k=i;k>temp;--k) fac1*=k;
  			for(k=i-temp;k>=1;--k) fac2*=k;
  			likelihood=(fac1/fac2)*pow(((double)a/i),a)*pow(((double)b/i),b);
  			a+=aprior; b+=bprior;
  			cout<<"Case "<<count<<": "<<line<<endl<<"Likelihood: "<<likelihood<<"\nBeta Prior: a = "<<aprior<<" b = "<<bprior<<endl;
  			cout<<"Beta Posterior: a = "<<a<<" b = "<<b<<endl;
  			cout<<"\n\n";
  			delete[] cstr;
    	}
    	myfile.close();
  	}
    return 0;
}
