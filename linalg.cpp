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
    for(int i = 0; i<matrix1.size();i++){
        for(int j = 0; j<matrix1[0].size();j++){
            matrix1[i][j]+=matrix2[i][j];
        }
    }
    return matrix1;
}

vvd matrix_sub(vvd matrix1, vvd matrix2){
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
