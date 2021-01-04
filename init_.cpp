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


vector<vector<double> > init_matrix(int size1, int size2){
    vector<vector<double> > matrix(size1);
    for(int i = 0;i<size1;i++){
        matrix[i]=vector<double>(size2);
        for (int j = 0; j<size2;j++){
            matrix[i][j]=norm(re);
        }
    }
    return matrix;
}

vector<vector<double> > ZEROmatrix(int size1, int size2){
    vector<vector<double> > matrix(size1);
    for(int i = 0;i<size1;i++){
        matrix[i]=vector<double>(size2);
        for (int j = 0; j<size2;j++){
            matrix[i][j]=0.0;
        }
    }
    return matrix;
}
vector<vector<double> > ONESmatrix(int size1, int size2){
    vector<vector<double> > matrix(size1);
    for(int i = 0;i<size1;i++){
        matrix[i]=vector<double>(size2);
        for (int j = 0; j<size2;j++){
            matrix[i][j]=1.0;
        }
    }
    return matrix;
}

vv3d init_3Dtensor(int size1, int size2, int size3){
    srand(time(0));
    vv3d matrix(size1);
    for(int i = 0;i<size1;i++){
        matrix[i]=vector<vector<double> >(size2);
        for (int j = 0; j<size2;j++){
            matrix[i][j]=vector<double>(size3);
            for (int k = 0; k<size3;k++){
                matrix[i][j][k]=norm(re);
            }
        }
    }
    return matrix;
}

vv3d init_3DZerotensor(int size1, int size2, int size3){
    srand(time(0));
    vv3d matrix(size1);
    for(int i = 0;i<size1;i++){
        matrix[i]=vector<vector<double> >(size2);
        for (int j = 0; j<size2;j++){
            matrix[i][j]=vector<double>(size3);
            for (int k = 0; k<size3;k++){
                matrix[i][j][k]=0;
            }
        }
    }
    return matrix;
}

map<string,vvd> tidyVars(string typeName,vvd f, vvd i , vvd o, vvd c, vvd y){
    map<string, vvd> parameters;
    assert(typeName == "weights" || typeName == "bias");
    if(typeName == "weights"){
        parameters.insert(pair<string,vvd>("Wf",f));
        parameters.insert(pair<string,vvd>("Wi",i));
        parameters.insert(pair<string,vvd>("Wo",o));
        parameters.insert(pair<string,vvd>("Wc",c));
        parameters.insert(pair<string,vvd>("Wy",y));
    }else{
        parameters.insert(pair<string,vvd>("bf",f));
        parameters.insert(pair<string,vvd>("bi",i));
        parameters.insert(pair<string,vvd>("bo",o));
        parameters.insert(pair<string,vvd>("bc",c));
        parameters.insert(pair<string,vvd>("by",y));
    }
    return parameters;

}

msv tidyGrads(vvd da_next, vvd dc_next ){
                    msv gradients;
                    gradients.insert(pair<string,vvd>("da_next",da_next));
                    gradients.insert(pair<string,vvd>("dc_next",dc_next));
                    return gradients;

                }
