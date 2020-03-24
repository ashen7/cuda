#include <stdio.h>
#include <time.h>

const int a_row = 2;
const int a_col = 3;
const int b_row = 3;
const int b_col = 4;

int main() {
    int start = clock();
    int a[a_row][a_col] = { 0 };
    int b[b_row][b_col] = { 0 };
    int c[a_row][b_col] = { 0 };

    for (int i = 0; i < a_row; i++) {
        for (int j = 0; j < a_col; j++) {
            a[i][j] = i * a_col + j;
        }
    }

    for (int i = 0; i < b_row; i++) {
        for (int j = 0; j < b_col; j++) {
            b[i][j] = i * b_col + j;
        }
    }

    for (int i = 0; i < a_row; i++) {
        for (int j = 0; j < b_col; j++) {
            for (int k = 0; k < a_col; k++) {
                c[i][j] += (a[i][k] * b[k][j]);
                printf("a[%d][%d]=%d, b[%d][%d]=%d, c[%d][%d]=%d\n", i, k, a[i][k], k, j, b[k][j], i, j, c[i][j]);
            }
        }
    }
    
    int end = clock();
    printf("\n程序耗时：%ds\n", (end - start) / 1000);

    return 0;
}
