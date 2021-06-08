#include<iostream>
#include<cmath>
#include<cstdio>
#include<cstring>
#include<vector>
#include<algorithm>
#include <fstream>
using namespace std;
int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}
void ReadMNIST(int NumberOfImages, int DataOfAnImage, vector < vector < double > > & arr, char s[]) {
    arr.resize(NumberOfImages, vector < double > (DataOfAnImage));
    ifstream file(s, ios::binary);
    if (file.is_open()) {
        int buf;
        int number_of_images=NumberOfImages;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char * ) & buf, sizeof(int));
        file.read((char * ) & buf, sizeof(int));
        if (DataOfAnImage != 1) {
            file.read((char * ) & n_rows, sizeof(n_rows));
            n_rows = ReverseInt(n_rows);
            file.read((char * ) & n_cols, sizeof(n_cols));
            n_cols = ReverseInt(n_cols);
        }
        for (int i = 0; i < number_of_images; ++i) {
            if (DataOfAnImage != 1) {
                for (int r = 0; r < n_rows; ++r) {
                    for (int c = 0; c < n_cols; ++c) {
                        unsigned char temp = 0;
                        file.read((char * ) & temp, sizeof(temp));
                        arr[i][(n_rows * r) + c] = (double) temp;
                    }
                }
            } else {
                unsigned char temp = 0;
                file.read((char * ) & temp, sizeof(temp));
                arr[i][0] = (double) temp;
            }
        }
        file.close();
    }
}

int main(int argc,char* argv[]) {
    char s1[] = "train-images.idx3-ubyte";
    char s2[] = "train-labels.idx1-ubyte";
    char s3[] = "test-images.idx3-ubyte";
    char s4[] = "test-labels.idx1-ubyte";
    vector < vector < double > > training_images; 
    vector < vector < double > > training_labels;
    vector < vector < double > > testing_images;
    vector < vector < double > > testing_labels;
	/*pixels: 784 = 28*28*/
    ReadMNIST(60000,784,training_images,s1); 
    ReadMNIST(60000,1,training_labels,s2); 
    ReadMNIST(10000,784,testing_images,s3); 
    ReadMNIST(10000,1,testing_labels,s4); 

    /*Calculating prior probabilities for classes 0-9*/
    int count[10],i,j,k;
    double Prior[10];
    for (int j=0;j<10;++j)
        	Prior[j]=0;
    for (int j=0;j<10;++j)
            count[j]=0;
    for (int i = 0; i < 60000; i++) {
        	count[(int)training_labels[i][0]]++;
    }	
    for (int i = 0; i <= 9; i++) {
        Prior[i] = (double) count[i] / 60000;
    }
	int toggle;
	toggle=atoi(argv[1]);
	if(toggle==0){
        double LUT[10][784][32], max = 0;        
        for (i = 0; i < 10; i++) {
            for (j = 0; j < 784; j++) {
                for (k = 0; k < 32; k++) {
                    LUT[i][j][k]=1;	
                }
            }
        }
		for (i = 0; i < 60000; i++) {
            for (j = 0; j < 784; j++) {
                LUT[(int)training_labels[i][0]][j][((int)training_images[i][j]>>3)]++;
            }
        }
        for (i = 0; i < 10; i++) {
            for (j = 0; j < 784; j++) {
                for (k = 0; k < 32; k++) {
                    LUT[i][j][k] = LUT[i][j][k] / (count[i]+2);	
                }
            }
        }

		long double posterior[10]; double marginal;
        int ans = 0, error = 0, min=0;
        for (i = 0; i < 10000; i++) {
        	marginal=0;
            for (j = 0; j < 10; j++) {
            	posterior[j]=0.0;
                for (k = 0; k < 784; k++) {
                    posterior[j]+=log(LUT[j][k][(int) testing_images[i][k]>>3]);
                }
                posterior[j]+=log(Prior[j]);
            }
            for (j = 0; j < 10; j++) {
                	marginal+=posterior[j];
                }
            cout<<"Posterior:\n";
            for (j = 0; j < 10; j++) {
                	posterior[j]=posterior[j]/marginal;
                	cout<<posterior[j]<<endl;
                }
            for (j = 0; j < 10; j++) {
                	if(posterior[j]<posterior[min])	min=j;
                }
            cout<<"Prediction: "<<min<<" Ans: "<<testing_labels[i][0]<<endl<<endl;
        	if(min!=testing_labels[i][0]) error++;
        }
    	cout << "Error rate: " << error / 100 << endl<<"\n\n";
    	for (i=0;i<10;++i){
    		for (j = 0; j < 784; j++) {
    			marginal=0;
                for (int k = 0; k < 16; k++) {
                    marginal+=LUT[i][j][k];	
                }
                if(marginal>0.5) cout<<"0";
                else cout<<"1";
                if((j+1)%28==0) cout<<endl;
            }
            cout<<"\n\n";
		}}
		else{
		
		double posterior[10]; double marginal;
        int error = 0, min=0;
        
		double mean[10][784], var[10][784];
		for(i=0;i<10;i++)
			for (j=0;j<784;++j)
				mean[i][j]=0.0;
		for(i=0;i<10;i++)
			for (j=0;j<784;++j)
				var[i][j]=100.0;
		for(i=0;i<60000;++i)
			for(j=0;j<784;++j)
				mean[(int)training_labels[i][0]][j]+=training_images[i][j];
		for(i=0;i<10;i++)
			for (j=0;j<784;++j)
				mean[i][j]/=count[i];
		for(i=0;i<60000;++i)
			for(j=0;j<784;++j)
				var[(int)training_labels[i][0]][j]+=(training_images[i][j]-mean[(int)training_labels[i][0]][j])*(training_images[i][j]-mean[(int)training_labels[i][0]][j]);
    	for(i=0;i<10;i++)
			for (j=0;j<784;++j)
				var[i][j]/=count[i]+100;
				///////////
		for (i = 0; i < 10000; i++) {
        	marginal=0;
            for (j = 0; j < 10; j++) {
            	posterior[j]=0.0;
                for (k = 0; k < 784; k++) {
                    posterior[j]+=-log(sqrt(2*3.14159265358979323846*var[j][k]))- (((testing_images[i][k]-mean[j][k])*(testing_images[i][k]-mean[j][k]))/(2*var[j][k]));
                }
                posterior[j]+=log(Prior[j]);
            }
            for (j = 0; j < 10; j++) {
                	marginal+=posterior[j];
                }
            //for (int j = 0; j < 10; j++) posterior[j]*=Prior[j];
            cout<<"Posterior:\n";
            for (j = 0; j < 10; j++) {
                	posterior[j]=posterior[j]/marginal;
                	cout<<posterior[j]<<endl;
                }
            for (j = 0; j < 10; j++) {
                	if(posterior[j]<posterior[min])	min=j;
                }
            cout<<"Prediction: "<<min<<" Ans: "<<(int)testing_labels[i][0]<<endl<<endl;
        	if(min!=testing_labels[i][0]) error++;
        }
    	cout << "Error rate: " << error / 100 << endl;
    	for (i=0;i<10;++i){
    		for (j = 0; j < 784; j++) {
                if(mean[i][j]<128) cout<<"0";
                else cout<<"1";
                if((j+1)%28==0) cout<<endl;
            }
            cout<<"\n\n";
		}}
	return 0;
}
