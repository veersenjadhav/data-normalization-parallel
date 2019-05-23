#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include <chrono>

#define ROWS 36634
#define FEATURES 14
#define DATASET "Dataset.csv"

using namespace std;
using namespace chrono;

int main(int argc, char const *argv[])
{

    string line, field;

    double* dataset = new double[ROWS * FEATURES];
    double* min = new double[FEATURES], *max = new double[FEATURES];
    double* normalized = new double[ROWS * FEATURES];

    ifstream in(DATASET);

    int value = 0;
    while ( getline(in,line) )
    {
        stringstream ss(line);

        while (getline(ss,field,',')) 
        {
            dataset[value] = (double)atof(field.c_str());
            value++;
        }
    }

    auto start = steady_clock::now();
    
    for(int j=0; j<FEATURES; j++)
    {
        double temp_min = dataset[j];
        for(int i=1; i<ROWS; i++)
        {
            if(dataset[i * FEATURES + j] < temp_min)
            {
                temp_min = dataset[i * FEATURES + j];
            }
        }
        min[j] = temp_min;
    }

    for(int j=0; j<FEATURES; j++)
    {
        double temp_max = dataset[j];
        for(int i=0; i<ROWS; i++)
        {
            if(dataset[i * FEATURES + j] > temp_max)
            {
                temp_max = dataset[i * FEATURES + j];
            }
        }
        max[j] = temp_max;
    }

    for(int i=0; i<ROWS; i++)
    {
        for(int j=0; j<FEATURES; j++)
        {
            normalized[i * FEATURES + j] = (dataset[i * FEATURES + j] - min[j]) / (max[j] - min[j]);
        }
    }

    auto end = steady_clock::now();

    /*for(int i=0; i<ROWS; i++)
    {
        for(int j=0; j<FEATURES; j++)
        {
            printf("%.8f \t",normalized[i * FEATURES + j]);
        }
        printf("\n");
    }*/

    cout << "\n Elapsed time in seconds : " 
        << chrono::duration_cast<chrono::milliseconds>(end - start).count() 
        << " millisec" << endl; 

    delete[] dataset, min, max, normalized;

    return 0;
}