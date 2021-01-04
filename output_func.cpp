#include <iostream>
#include <map>
#include <vector>
#include<time.h>
#include<string>
#include<stdlib.h>
#include<random>
#include<math.h>
#include <fstream>
using namespace std;
typedef long long ll;
typedef long double ld;
typedef vector<vector<double> > vvd;
typedef vector<vector<vector<double> > > vv3d;
typedef map<string,vvd> msv;

//print matrix
void printMatrix(vvd matrix){
    for (int i = 0; i<matrix.size();i++){
        for (int j = 0;j<matrix[i].size();j++){
            cout<<matrix[i][j]<<" ";
        }
        cout<<endl;
    }

}

void print3Dtensor(vv3d tensor){
    for (int i = 0; i<tensor.size();i++){
        for (int j = 0;j<tensor[i].size();j++){
            for (int k =0; k<tensor[i][j].size();k++){
                cout<<tensor[i][j][k]<<" ";
            }
            cout<<"  "<<endl;
        }
        cout<<"    "<<endl;
    }
}

void writeMatrix(ofstream &file,vvd matrix){
    for (int i = 0; i<matrix.size();i++){
        for (int j = 0;j<matrix[i].size();j++){
            file<<matrix[i][j]<<" ";
        }
        file<<endl;
    }
}

void writeTensor(ofstream &file ,vv3d tensor){
    for (int i = 0; i<tensor.size();i++){
        for (int j = 0;j<tensor[i].size();j++){
            for (int k =0; k<tensor[i][j].size();k++){
                file<<tensor[i][j][k]<<" ";
            }
            file<<"  "<<endl;
        }
        file<<"    "<<endl;
    }
}
