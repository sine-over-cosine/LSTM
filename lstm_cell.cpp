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

class LSTM{
    public :
    // Actual data
    vv3d data;
    vvd initial_state;
    vvd x_target_time;
    vv3d da;
    vvd da_target_time;
    vvd da_next;

    vv3d a, c, y;
    //To initialise
    vvd xt;
    vvd a_prev;
    vvd c_prev;
    msv weights;
    msv bias;

    //For cache
    vvd a_next;
    vvd c_next;
    vvd yt_pred;
    vvd ft, it, cct, ot;
    //For massive cacheing for backpropagation
    //GATE RELATED GRADIENTS
    vvd dot,dcct,dit,dft;
    /*int time t -> map<string -> vvd >*/
    map<int,msv> cache;
    ofstream file;
    LSTM(
         msv weights_param,
         msv bias_param,
         vv3d main_data,
         vvd a0,
         vv3d da0){
             weights = weights_param;
             bias = bias_param;
             data = main_data;
             initial_state = a0;
             da = da0;
         }
    
    void call_me(){
        cout<<"OK"<<endl;
    }

    void lstm_cell_forward(){
        vvd Wf = this->weights["Wf"];
        vvd Wi = this->weights["Wi"];
        vvd Wc = this->weights["Wc"];
        vvd Wo = this->weights["Wo"];
        vvd Wy = this->weights["Wy"];
        vvd bf = this->bias["bf"];
        vvd bi = this->bias["bi"];
        vvd bc = this->bias["bc"];
        vvd bo = this->bias["bo"];
        vvd by = this->bias["by"]; 
        
        vvd concat = concatenate(this->a_next,this->x_target_time,"vertical");
        
        vvd bf_b = broadcast(bf, "horizontal",concat[0].size()-bf[0].size());
        vvd bi_b = broadcast(bi, "horizontal",concat[0].size()-bi[0].size());
        vvd bc_b = broadcast(bc, "horizontal",concat[0].size()-bc[0].size());
        vvd bo_b = broadcast(bo, "horizontal",concat[0].size()-bo[0].size());
        vvd by_b = broadcast(by, "horizontal",concat[0].size()-by[0].size());
        
        this->ft = sigmoid(matrix_add(matrix_mul(Wf,concat,false),bf_b));
        this->it = sigmoid(matrix_add(matrix_mul(Wi,concat,false),bi_b));
        this->cct = tanh_activation(matrix_add(matrix_mul(Wc,concat,false), bc_b));
        this->c_next = matrix_add(matrix_mul(ft,this->c_next,true),matrix_mul(this->it,this->cct,true));
        this->ot = sigmoid(matrix_add(bo_b,matrix_mul(Wo,concat,false)));
        this->a_next = matrix_mul(this->ot,tanh_activation(this->c_next),true);
        this->yt_pred = softmax(matrix_add(by_b,matrix_mul(Wy,this->a_next,false)));
        
        
    } 

    void lstm_forward(){

        int n_x = data.size();
        int m = data[0].size();
        int T_x = data[0][0].size();
        int n_y = weights["Wy"].size();
        int n_a = weights["Wy"][0].size();

        this->a_next = this->initial_state;
        this->c_next = ZEROmatrix(this->initial_state.size(),this->initial_state[0].size());
        
        this->a = init_3DZerotensor(n_a, m, T_x);
        this->c = init_3DZerotensor(n_a, m, T_x);
        this->y = init_3DZerotensor(n_y, m, T_x);
        
        for(int i = 0; i < T_x; i++){
            file<<"For time: "<<i<<"\n"<<endl;
            this->x_target_time = slice(data, 2, i);
            file<<"x_t"<<endl;
            writeMatrix(file,this->x_target_time);
            this->cache[i]["a_prev"]=this->a_next;
            file<<"a_prev"<<endl;
            writeMatrix(file,this->cache[i]["a_prev"]);
            this->cache[i]["c_prev"]=this->c_next; 
            file<<"c_prev"<<endl;
            writeMatrix(file,this->cache[i]["c_prev"]);
            lstm_cell_forward();
            this->cache[i]["xt"]=this->x_target_time;
            file<<"xt"<<endl;
            writeMatrix(file,this->cache[i]["xt"]);
            this->cache[i]["ft"]=this->ft;
            file<<"ft"<<endl;
            writeMatrix(file,this->cache[i]["ft"]);
            this->cache[i]["it"]=this->it;
            file<<"it"<<endl;
            writeMatrix(file,this->cache[i]["it"]);
            this->cache[i]["cct"]=this->cct;
            file<<"cct"<<endl;
            writeMatrix(file,this->cache[i]["cct"]);
            this->cache[i]["c_next"]=this->c_next;
            file<<"c_next"<<endl;
            writeMatrix(file,this->cache[i]["c_next"]);
            this->cache[i]["ot"]=this->ot;
            file<<"ot"<<endl;
            writeMatrix(file,this->cache[i]["ot"]);
            this->cache[i]["a_next"]=this->a_next;
            file<<"a_next"<<endl;
            writeMatrix(file,this->cache[i]["a_next"]);
            this->cache[i]["yt_pred"]=this->yt_pred;
            file<<"yt_pred"<<endl;
            writeMatrix(file,this->cache[i]["yt_pred"]);
            fill3Dwith2D(this->a,this->a_next,2,i);
            fill3Dwith2D(this->y,this->yt_pred,2,i);
            fill3Dwith2D(this->c,this->c_next,2,i);

            
        }
        file<<"a"<<endl;
        writeTensor(file,this->a);
        file<<"c"<<endl;
        writeTensor(file,this->c);
        file<<"y"<<endl;
        writeTensor(file,this->y);
    }

    void lstm_cell_backward(int time){
        int n_x = this->cache[time]["xt"].size();
        int m = this->cache[time]["xt"][0].size();
        int n_a = this->cache[time]["a_next"].size();

        dot = matrix_mul(matrix_mul(tanh_activation(this->cache[time]["c_next"]),matrix_sub(this->cache[time]["ot"],matrix_mul(this->cache[time]["ot"],this->cache[time]["ot"],true)),true),this->da_target_time,true);
        this->cache[time]["dot"]=dot;
        vvd common_multiple = matrix_add(this->cache[time]["dc_prevt"],matrix_mul(this->da_target_time,matrix_mul(oneMinusXSquared(tanh_activation(this->cache[time]["c_next"])),this->cache[time]["ot"],true),true));
        dcct = matrix_mul(matrix_mul(common_multiple,oneMinusXSquared(this->cache[time]["cct"]),true),this->cache[time]["it"],true);
        this->cache[time]["dcct"]=dcct;
        dit = matrix_mul(matrix_mul(matrix_sub(this->cache[time]["it"],matrix_mul(this->cache[time]["it"],this->cache[time]["it"],true)),this->cache[time]["cct"],true),common_multiple,true);
        this->cache[time]["dit"]=dit;
        dft = matrix_mul(matrix_mul(matrix_sub(this->cache[time]["ft"],matrix_mul(this->cache[time]["ft"],this->cache[time]["ft"],true)),this->cache[time]["c_prev"],true),common_multiple,true);
        this->cache[time]["dft"]=dft;

        vvd dWf = matrix_mul(dft,concatenate(transpose(this->cache[time]["a_prev"]),transpose(this->cache[time]["xt"]),"horizontal"),false);
        vvd dWi = matrix_mul(dit,concatenate(transpose(this->cache[time]["a_prev"]),transpose(this->cache[time]["xt"]),"horizontal"),false);
        vvd dWc = matrix_mul(dcct,concatenate(transpose(this->cache[time]["a_prev"]),transpose(this->cache[time]["xt"]),"horizontal"),false);
        vvd dWo = matrix_mul(dot,concatenate(transpose(this->cache[time]["a_prev"]),transpose(this->cache[time]["xt"]),"horizontal"),false);
        this->cache[time]["dWf"]=dWf;
        this->cache[time]["dWi"]=dWi;
        this->cache[time]["dWc"]=dWc;
        this->cache[time]["dWo"]=dWo;

        vvd dbf = sum2D(dft,"horizontal");
        vvd dbi = sum2D(dit,"horizontal");
        vvd dbc = sum2D(dcct,"horizontal");
        vvd dbo = sum2D(dot,"horizontal");
        this->cache[time]["dbf"]=dbf;
        this->cache[time]["dbi"]=dbi;
        this->cache[time]["dbc"]=dbc;
        this->cache[time]["dbo"]=dbo;

        vvd da_prev = matrix_add(matrix_mul(transpose(slice(this->weights["Wo"],2,n_a,"first")),dot,false),matrix_add(matrix_mul(transpose(slice(this->weights["Wi"],2,n_a,"first")),dit,false),matrix_add(matrix_mul(transpose(slice(this->weights["Wc"],2,n_a,"first")),dcct,false),matrix_mul(transpose(slice(this->weights["Wf"],2,n_a,"first")),dft,false))));
        vvd dc_prev = matrix_mul(common_multiple,this->cache[time]["ft"],true);
        vvd dxt = matrix_add(matrix_mul(transpose(slice(this->weights["Wo"],2,n_x,"final")),dot,false),matrix_add(matrix_mul(transpose(slice(this->weights["Wi"],2,n_x,"final")),dit,false),matrix_add(matrix_mul(transpose(slice(this->weights["Wc"],2,n_x,"final")),dcct,false),matrix_mul(transpose(slice(this->weights["Wf"],2,n_x,"final")),dft,false))));
        this->cache[time]["da_prev"]=da_prev;
        this->cache[time]["dc_prev"]=dc_prev;
        this->cache[time]["dxt"]=dxt;
        
        
    }

    void lstm_backward(){
        int n_a = this->da.size();
        int m = this->da[0].size();
        int T_x = this->da[0][0].size();
        int n_x = this->cache[0]["xt"].size();
        
        vv3d dx = init_3DZerotensor(n_x,m,T_x);
        vvd da0 = ZEROmatrix(n_a,m);
        vvd da_prevt = ZEROmatrix(n_a,m);
        vvd dc_prevt = ZEROmatrix(n_a,m);
        vvd dWf = ZEROmatrix(n_a,n_a+n_x);
        vvd dWi = ZEROmatrix(n_a,n_a+n_x);
        vvd dWc = ZEROmatrix(n_a,n_a+n_x);
        vvd dWo = ZEROmatrix(n_a,n_a+n_x);
        vvd dbf = ZEROmatrix(n_a,1);
        vvd dbi = ZEROmatrix(n_a,1);
        vvd dbc = ZEROmatrix(n_a,1);
        vvd dbo = ZEROmatrix(n_a,1);
        
        
        for(int t = T_x - 1; t >= 0; t--){
            this->cache[t]["dc_prevt"]=dc_prevt;
            this->da_target_time = matrix_add(slice(da,2,t),da_prevt);
            lstm_cell_backward(t);
            fill3Dwith2D(dx,this->cache[t]["dxt"],2,t);
            dWf = matrix_add(dWf,this->cache[t]["dWf"]);
            dWi = matrix_add(dWf,this->cache[t]["dWi"]);
            dWc = matrix_add(dWf,this->cache[t]["dWc"]);
            dWo = matrix_add(dWf,this->cache[t]["dWo"]);
            dbf = matrix_add(dbf,this->cache[t]["dbf"]);
            dbi = matrix_add(dbf,this->cache[t]["dbi"]);
            dbc = matrix_add(dbf,this->cache[t]["dbc"]);
            dbo = matrix_add(dbf,this->cache[t]["dbo"]);
        }
        this->cache[0]["da0"]=this->cache[0]["da_prev"];
    }

    void set_up(){
        file.open("Report.txt");
        lstm_forward();
        lstm_backward();
        file.close();
        
    }

};
