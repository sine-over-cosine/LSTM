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

//
vector<vector<double> > init_matrix(int size1, int size2);



//LINEAR ALGEBRA OPERATIONS
vvd matrix_add(vvd matrix1, vvd matrix2){
    assert(matrix1.size()==matrix2.size());
    assert(matrix1[0].size()==matrix2[0].size());
    for(int i = 0; i<matrix1.size();i++){
        for(int j = 0; j<matrix1[0].size();j++){
            matrix1[i][j]+=matrix2[i][j];
        }
    }
    return matrix1;
}

vvd matrix_sub(vvd matrix1, vvd matrix2){
    assert(matrix1.size()==matrix2.size());
    assert(matrix1[0].size()==matrix2[0].size());
    for(int i = 0; i<matrix1.size();i++){
        for(int j = 0; j<matrix1[0].size();j++){
            matrix1[i][j]-=matrix2[i][j];
        }
    }
    return matrix1;
}

vvd transpose(vvd matrix){
    vvd new_matrix = init_matrix(matrix[0].size(),matrix.size());
    for (int i = 0; i<new_matrix.size();i++){
        for (int j = 0; j<new_matrix[0].size();j++){
            new_matrix[i][j]=matrix[j][i];
        }
    }
    return new_matrix;
}

vvd matrix_mul(vvd matrix1, vvd matrix2, bool element_wise){
    if(element_wise){
        assert(matrix1.size()==matrix2.size());
        assert(matrix1[0].size()==matrix2[0].size());
        for(int i = 0; i<matrix1.size();i++){
            for(int j = 0; j<matrix1[0].size();j++){
                matrix1[i][j]*=matrix2[i][j];
            }
        }
        return matrix1;
    }else{
        assert(matrix1[0].size()==matrix2.size());
        vvd new_matrix(matrix1.size(),vector<double>(matrix2[0].size(),0));
        for(int i = 0 ; i<matrix1.size(); i++){
            for (int j = 0;j<matrix2[0].size();j++){
                new_matrix[i][j]=0;
                for (int k = 0;k<matrix2.size();k++){
                    new_matrix[i][j]+=(matrix1[i][k]*matrix2[k][j]);
                }
            }
        }
        return new_matrix;
    }
}

void shape(vvd matrix1){
    cout<<"["<<matrix1.size()<<","<<matrix1[0].size()<<"]"<<endl;
}

vvd broadcast(vvd matrix, string axis, int dims){
    assert(axis == "vertical" || axis == "horizontal");
    assert(dims > 0);
    if (axis == "horizontal"){
        int left = dims - matrix[0].size();
        for(int i = 0; i < matrix.size(); i++){
            for(int j = 0; j <= left; j++){
                matrix[i].push_back(matrix[i][j]);
            }
        }
    }else{
        int left = dims - matrix.size();
        for(int i = 0; i <= left; i++){
            matrix.push_back(matrix[i]);
        }
    }
    return matrix;
}
