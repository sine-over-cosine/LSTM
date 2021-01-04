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

//Prototype
vvd init_matrix(int size1, int size2);
vv3d init_3Dtensor(int size1, int size2, int size3);
vvd ZEROmatrix(int size1, int size2);
vvd slice(vv3d tensor, int dim, int index );
vv3d init_3DZerotensor(int size1, int size2, int size3);
void fill3Dwith2D(vv3d &tensor, vvd &matrix, int dims, int index);
vvd sum2D(vvd matrix, string axis);

void ok(){
    cout<<"OK!"<<endl;
}
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
//MATHS OPERATIONS

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
//INITIALISE FUNCTIONS
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

void fill3Dwith2D(vv3d &tensor, vvd &matrix, int dims, int index){
    //To be built in progress
    assert(dims == 1 || dims == 2 || dims == 0);
    if(dims == 2){
        for(int i = 0 ; i < matrix.size(); i++){
            for(int j = 0; j < matrix[0].size(); j++){
                tensor[i][j][index]=matrix[i][j];
            }
        }
    }
}

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


