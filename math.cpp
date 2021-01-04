#include <iostream>
#include <map>
#include <vector>
#include<time.h>
#include<string>
#include<stdlib.h>
#include<random>
#include<math.h>

using namespace std;
typedef long long ll;
typedef long double ld;
typedef vector<vector<double> > vvd;
typedef vector<vector<vector<double> > > vv3d;
typedef map<string,vvd> msv;

normal_distribution<long double> norm(0.0,1.0);
default_random_engine re;

vvd oneMinusXSquared(vvd matrix){
    vector<vector<double> > ones(matrix.size());
    for(int i = 0;i<matrix.size();i++){
        ones[i]=vector<double>(matrix[0].size());
        for (int j = 0; j<matrix[0].size();j++){
            ones[i][j]=1.0;
        }
    }
    matrix = matrix_sub(ones,matrix_mul(matrix,matrix,true));
    return matrix;
}

vvd concatenate(vvd matrix1, vvd matrix2, string axis){
    if(axis == "horizontal"){
        for(int i = 0; i<matrix1.size();i++){
            matrix1[i].insert(matrix1[i].end(),matrix2[i].begin(),matrix2[i].end());
        }
        return matrix1;
    }
    else if(axis == "vertical"){
        matrix1.insert(matrix1.end(),matrix2.begin(),matrix2.end());
        }
        return matrix1;
    }

double sigmoid(double x){
    return 1.0/(1+exp(-1 * x));
}

vvd sigmoid(vvd matrix){
    for(int i = 0; i<matrix.size(); i++){
        for (int j = 0; j<matrix[0].size();j++){
            matrix[i][j]=sigmoid(matrix[i][j]);
        }
    }
    return matrix;
}

vvd tanh_activation(vvd matrix){
    for(int i = 0; i<matrix.size(); i++){
        for (int j = 0; j<matrix[0].size();j++){
            matrix[i][j]=tanh(matrix[i][j]);
        }
    }
    return matrix;
}

vvd softmax(vvd matrix){
    //Exponentiate every entry; collecting sum of exponentiated items meanwhile
    double denominators[matrix[0].size()];
    for(int a = 0; a< matrix[0].size();a++){
        denominators[a]=0.0;
    }
    for(int i = 0; i< matrix.size();i++){
        for(int j = 0; j<matrix[0].size(); j++){
            matrix[i][j]=exp(matrix[i][j]);
            denominators[j]+=matrix[i][j];
        }
    }
    for(int i = 0; i < matrix.size(); i++){
        for(int j = 0; j< matrix[0].size(); j++){
            matrix[i][j]/=denominators[j];
        }
    }
    return matrix;
}

vvd slice(vvd matrix, int dim , int index, string status){
    assert(status == "first" || status == "final");
    assert(dim == 1 || dim == 2); //1 is row; 2 is column
    vvd output;
    if(dim == 1){
        if(status=="first"){
            for(int i = 0; i < index ; i++){
                output.push_back(matrix[i]);
            }
            return output;
        }
        else{
            for(int i = index - 1; i < matrix.size(); i++ ){
                output.push_back(matrix[i]);
            }
            return output;
        }
    }else{
        if(status=="first"){
            for(int i = 0; i < matrix.size(); i++){
                output.push_back(vector<double>(index));
                for(int j = 0; j < index; j++){
                    output[i][j]=matrix[i][j];
                }
            }
            return output;
        }else{
            for(int i = 0; i < matrix.size(); i++){
                output.push_back(vector<double>(index));
                for(int j = 0 ; j < index; j++){
                    output[i][j]=matrix[i][j + matrix[0].size() - index];
                }
            }
            return output;
        }
    }
}

vvd slice(vv3d tensor, int dim, int index ){
    /*
    dimension is either [0,1,2];
    */
   vvd matrix(tensor.size());
   assert(dim == 1 || dim ==2 || dim == 0);
    switch(dim){
        case 0:
            return tensor[index];
        case 1:
            for (int i = 0; i< tensor.size();i++){
                matrix.push_back(tensor[i][index]);
            }
            return matrix;
        case 2:
            for(int i = 0; i< tensor.size(); i++){
                matrix[i]= vector<double>(tensor[0].size());
                for(int j = 0; j < tensor[0].size(); j++){
                    matrix[i][j]=tensor[i][j][index];
                }
            }
            return matrix;
    }
}

vvd sum2D(vvd matrix, string axis){
    assert(axis == "horizontal" || axis == "vertical");
    vvd output;
    if(axis == "horizontal"){
        output = ZEROmatrix(matrix.size(),1);
        //SUM over all horizontals
        for(int i = 0; i<matrix.size();i++){
            for(int j = 0; j<matrix[0].size(); j++){
                output[i][0]+=matrix[i][j];
            }
        }
    }else if(axis=="vertical"){
            output = ZEROmatrix(1,matrix[0].size());
            for(int i = 0; i < matrix.size(); i++){
                for(int j = 0; j < matrix[0].size(); j++){
                    output[0][j]+=matrix[i][j];
                }
            }
    }
    return output;
}
