#include <iostream>

using std::cout;
using std::endl;


void print2DArray(int** arr, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << arr[i][j] << '\t';
        }
        cout << endl;
    }
}

int main() {
    const int N = 10;

    int** x = new int*[N];
    for (int i=0; i<N; i++) {
        x[i] = new int[N]; 
    }

    print2DArray(x, N, N);


    // Free memory
    for (int i = 0; i < N; i++) {
        delete[] x[i];
    }
    delete[] x;


    return 0;
}