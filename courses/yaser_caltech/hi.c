#include<stdio.h>
#include<stdlib.h>


float one_run(float *out) {
    int t;
    int N = 1000;
    int m = 10;
    int i = 0;
    int j = 0;
    int randindex = 0;
    int first_count = 0;
    int rand_count = 0;
    int min_count = m;
    int head_count = 0;
    srand((unsigned) time(&t));
    randindex = rand() % N;

    for (i=0; i<N; i++) {
        head_count = 0;
        for(j=0; j<m; j++){
            if (rand() % 100 >= 50){
               head_count ++; 
            }
        }
        if (i == 0) first_count = head_count;
        if (i == randindex) rand_count = head_count;
        if (min_count > head_count) min_count = head_count;
    }
    out[0] = (float)first_count / m;
    out[1] = (float)rand_count / m;
    out[2] = (float)min_count / m;
}

int main(){
    float out[3] = {0, 0, 0};
    int N = 10000;
    float v1_sum=0, vrand_sum=0, vmin_sum=0;
    int i = 0;
    for(;i < N; i++){
        one_run(out);
        v1_sum += out[0];
        vrand_sum += out[1];
        vmin_sum += out[2];
    }
    printf("%f %f %f\n", v1_sum / N, vrand_sum / N, vmin_sum / N);
    return 0;
}
