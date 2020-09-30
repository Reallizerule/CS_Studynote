//
//  main.cpp
//  滑雪
//
//  Created by Barbra on 2020/9/17.
//  Copyright © 2020 Barbra. All rights reserved.
//

#include <iostream>
#include <vector>
using namespace std;
int nmax = 1;
void cal ( int m , int n , int re ,vector<vector<int>>& num){
    if ( m != 0 && num[m-1][n] < num[m][n] ){
        int t = re + 1;
        if ( t > nmax ){
            nmax = t;
        }
        cal ( m-1 , n , t , num);
    }
    if ( m != num.size()-1 && num[m+1][n] < num[m][n] ){
        int t = re + 1;
        if ( t > nmax ){
            nmax = t;
        }
        cal ( m+1 , n , t , num );
    }
    if ( n != 0 && num[m][n-1] < num[m][n] ){
        int t = re + 1;
        if ( t > nmax ){
            nmax = t;
        }
        cal ( m , n-1 , t , num );
    }
    if ( n != num[0].size()-1 && num[m][n+1] < num[m][n]){
        int t = re + 1;
        if ( t > nmax ){
            nmax = t;
        }
        cal ( m , n+1 , t , num );
    }
}
int main(int argc, const char * argv[]) {
    int n1,n2;
    cin >> n1 >> n2;
    if ( n1 == 0 || n2 == 0 ){
        cout << 0 << endl;
        return 0;
    }
    vector<vector<int>> num(n1,vector<int>(n2,0));
    for ( int i = 0 ; i < n1 ; i++ ){
        for ( int j = 0 ; j < n2 ; j++ ){
            cin >> num[i][j];
        }
    }
    for ( int i = 0 ; i < n1 ; i++ ){
        for ( int j = 0 ; j < n2 ; j++ ){
            cal( i , j , 1 , num );
        }
    }
    cout << nmax << endl;
    return 0;
}
