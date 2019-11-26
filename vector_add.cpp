#include <stdio.h>
#include <time.h>

const int N = 512 * 1024;

int main() {
    int start = clock();
    float a[N] = { 0.0 };
    float b[N] = { 0.0 };
    float c[N] = { 0.0 };

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
        c[i] = a[i] + b[i];
    }
    
    for (int i = 0; i < N; i++) {
        printf("%.0f + %.0f = %.0f\t", a[i], b[i], c[i]);
        if ((i + 1) % 10 == 0)
            printf("\n");
    }
    int end = clock();
    printf("\n程序耗时：%ds\n", (end - start) / 1000);

    return 0;
}
