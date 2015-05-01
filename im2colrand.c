#include "mex.h"
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize h, w;
    mwSize bsizeX, bsizeY, psize, numPatches;
    mwIndex a, n, x, y, i;
    
    double *bsize, *X, *Y;
    
    // Extract image dimensions
    h = mxGetM(prhs[0]);
    w = mxGetN(prhs[0]);
    
    // Get the pointer to the data
    X = mxGetPr(prhs[0]);
    bsize = mxGetPr(prhs[1]);
    numPatches = mxGetScalar(prhs[2]);
    
    // Copy the block size parameters
    bsizeX = bsize[0];
    bsizeY = bsize[1];
    psize = bsizeX * bsizeY;
    
    // Create the output matrix
    plhs[0] = mxCreateDoubleMatrix(psize, numPatches, mxREAL);
    Y = mxGetPr(plhs[0]);
    
    // Seed the random number generator
    srand(0);
    
    for (n = 0; n < numPatches; n++)
    {
        x = rand() % (w - bsizeX + 1);
        y = rand() % (h - bsizeY + 1);
        
        a = 0;
        for (i = 0; i < bsizeX; i++)
        {
            memcpy(&Y[psize*n+a], &X[h*(x+i)+y], bsizeY * sizeof(double));
            a += bsizeY;
        }
    }
}