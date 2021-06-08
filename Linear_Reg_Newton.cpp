#include <iostream>
#include<sstream>
#include <fstream>
#include<string>
#include<cstdlib>
#include<ctime>
#include <cmath>
using namespace std;
void AtA(double a[][50],double c[][50],double b[][50],int m,int n,int q)
{
	int i,j,k;
    for(i = 0; i < n; i++)
        for(j = 0; j < q; j++)
        {	b[i][j]=0.0;	
           	for(k = 0; k < m; ++k)
            	b[i][j] += a[k][i] * c[k][j];}
}
void choles(double a[][50], double L[][50], int n) 
{
   int i = 0, j = 0, k = 0;
   for (int i = 0; i < n; i++)
        for (int j = 0; j < (i+1); j++) {
            double s = 0;
        	for (int k = 0; k < j; k++)
                s += L[i][k] * L[j][k];
            L[i][j] = (i == j) ?sqrt(a[i][i] - s) :(1.0 / L[j][j] * (a[i][j] - s));
        }
}
void L_1(double a[][50],double b[][50],int n){
	int i,j,k;
	for(i=0;i<n;++i)
		for(j=0;j<n;++j)
			b[i][j]=0.0;
	for(k=0;k<n;++k)
	{
		b[k][k]=1.0/a[k][k];
		for(i=k+1;i<=n;++i)
		{
			for(j=k;j<i;++j)	
				b[i][k]+=-(a[i][j]*b[j][k]);
			b[i][k]/=a[i][i];					
		}
	}
}
int main(int argc,char *argv[]) {
    int n=atoi(argv[2]),i,j=0,k,l=100,m=100;
    double x[100][50],y[100],temp,tr[50][50],u[50][50],tempo[50],param[50],check[100],error,lambda,tem[50],epsilon=20;
    double param_new[50],norm;
    stringstream parse(argv[3]); 
    parse>>lambda;
    srand (time(NULL));
    ifstream inFile;
    ofstream will;
    string line,address=argv[1];
    string value;
    if (argc!=4)
    	{
    		cout<<"Not enough arguments !!";
    		return 0;
		}
    inFile.open(address.c_str());
    if (!inFile) {
        cout << "Unable to open file";
        exit(1); 
    }
    i=0;
    while(getline(inFile,line))
	{
    	stringstream   linestream(line);
    	while(getline(linestream,value,','))
    	{
    		stringstream parse(value); 
    		parse>>temp;
    		if(!(i&1))	
        		x[i>>1][1]=temp;
        	else
        		y[i>>1]=temp;
        	i++;
    	}
    	j++;
	}l=j;
    for (i=0;i<j;++i) {
    	for(k=0;k<n;++k)
    	x[i][k]=pow(x[i][1],k);
	}
	// preprocessing done
	
	AtA(x,x,tr,j,n,n);
	for(i=0;i<n;++i) tr[i][i]+=lambda;
	choles(tr,u,n);
	
    L_1(u,tr,n);
    AtA(tr,tr,u,n,n,n);
    for(i = 0; i < n; i++)
    {	tempo[i]=0.0;
        for(j = 0; j < l; j++)
        {		
		   	tempo[i] += x[j][i] * y[j];}}
	
	for(i = 0; i < n; ++i)
	{	param[i]=0.0;
        for(j = 0; j < n; ++j)
        {	
            param[i] += u[i][j] * tempo[j];
    	}}
	cout.precision(14);
	error=0.0;
    cout<<"\n\n\n\nRegularized Least Squares Estimation:\n-------------------------------------\n";
	cout<<"Best fitted line:\n";
	cout<<param[0]<<" + ";
    for(i=1;i<n-1;++i) cout<<param[i]<<" X^"<<i<<" + ";
    cout<<param[n-1]<<" X^"<<n-1<<endl;
    for(i=0;i<l;++i) {
    	check[i]=0.0;
    	for(j=0;j<n;++j) check[i]+=x[i][j]*param[j];
    	error+=(y[i]-check[i])*(y[i]-check[i]);
	}
	cout<<"\nError: "<<error<<endl;
	
	will.open ("parameter.txt");
	will << address<<endl;
  	for(i=0;i<l;++i) will << check[i]<<"\n";
  	will.close();
	//Newton's method
	for(i=0;i<n;++i) param[i]=rand()%10;
	while(epsilon>0.000001)
	{
		norm=0.0;	
		AtA(x,x,tr,l,n,n);
		for(i = 0; i < n; i++)
    	{	tempo[i]=0.0;
        	for(j = 0; j < l; j++)
        	{		
		   		tempo[i] += x[j][i] * y[j];}}
		for(i=0;i<n;++i)
		{
			tem[i]=0.0;
			for(j=0;j<n;++j)
			{
				tem[i] += tr[i][j] * param[j];
			}
			tem[i]-=tempo[i];
			tem[i]*=2;
		}
		cout<<endl;
			for(i=0;i<n;++i)
				for(j=0;j<n;++j)
					tr[i][j]*=2;
			for(i=0;i<n;++i)
				for(j=0;j<n;++j)
					u[i][j]*=0.0;
		choles(tr,u,n);
		L_1(u,tr,n);
    	AtA(tr,tr,u,n,n,n);
    	for(i=0;i<n;++i)
		{
			tempo[i]=0.0;
			for(j=0;j<n;++j)
			{
				tempo[i] += u[i][j] * tem[j];
			}
			param[i]-=tempo[i];
		}
		AtA(x,x,tr,l,n,n);
		for(i = 0; i < n; i++)
    	{	tempo[i]=0.0;
        	for(j = 0; j < l; j++)
        	{		
		   		tempo[i] += x[j][i] * y[j];}}
		for(i=0;i<n;++i)
		{
			tem[i]=0.0;
			for(j=0;j<n;++j)
			{
				tem[i] += tr[i][j] * param[j];
			}
			tem[i]-=tempo[i];
			tem[i]*=2;
		}
		for(i=0;i<n;++i) 
		{
			norm+=tem[i];
			epsilon=sqrt(norm);
		}
	}
	cout<<"Newton's method':\n-----------------\n";
	cout<<"Best fitted line:\n";
	cout<<param[0]<<" + ";
    for(i=1;i<n-1;++i) cout<<param[i]<<" X^"<<i<<" + ";
    cout<<param[n-1]<<" X^"<<n-1<<endl;
    error=0.0;
    for(i=0;i<l;++i) {
    	check[i]=0.0;
    	for(j=0;j<n;++j) check[i]+=x[i][j]*param[j];
    	error+=(y[i]-check[i])*(y[i]-check[i]);
	}
	cout<<"\nError: "<<error<<"\n\n\n\n\n\n";
	will.open ("newton.txt");
  	for(i=0;i<l;++i) will << check[i]<<"\n";
  	will.close();
    return 0;
}
