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

normal_distribution<long double> norm(0.0,1.0);
default_random_engine re;


int main(){
    vv3d x = init_3Dtensor(3,10,7);
    vvd a0 = init_matrix(5,10);
    vvd Wf = init_matrix(5, 5+3);
    vvd bf = init_matrix(5,1);
    vvd Wi = init_matrix(5, 5+3);
    vvd bi = init_matrix(5,1);
    vvd Wo = init_matrix(5, 5+3);
    vvd bo = init_matrix(5,1);
    vvd Wc = init_matrix(5, 5+3);
    vvd bc = init_matrix(5,1);
    vvd Wy = init_matrix(2,5);
    vvd by = init_matrix(2,1);
    vv3d da = init_3Dtensor(5,10,4);
    msv weights =  tidyVars("weights",Wf,Wi,Wo,Wc,Wy);
    msv bias = tidyVars("bias",bf,bi,bo,bc,by);
    LSTM test(weights, bias, x, a0, da);
    test.set_up();
    
    return 0;
}
