
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "MRC.h"
/*
 Read the MRC file
*/

void MRC::EMRead(string file)
{
    fstream inputFile(file.c_str(), ios::in|ios::binary);
    if(!inputFile)
    {
       // cerr << "Error: " << strerror(errno);
        exit(1);
    }

    inputFile.read(reinterpret_cast<char *>(&nx), sizeof(nx));
    inputFile.read(reinterpret_cast<char *>(&ny), sizeof(ny));
    inputFile.read(reinterpret_cast<char *>(&nz), sizeof(nz));
    inputFile.read(reinterpret_cast<char *>(&mode), sizeof(mode));
    inputFile.read(reinterpret_cast<char *>(&nxStart), sizeof(nxStart));
    inputFile.read(reinterpret_cast<char *>(&nyStart), sizeof(nyStart));
    inputFile.read(reinterpret_cast<char *>(&nzStart), sizeof(nzStart));
    inputFile.read(reinterpret_cast<char *>(&mx), sizeof(mx));
    inputFile.read(reinterpret_cast<char *>(&my), sizeof(my));
    inputFile.read(reinterpret_cast<char *>(&mz), sizeof(mz));
    inputFile.read(reinterpret_cast<char *>(&xLength), sizeof(xLength));
    inputFile.read(reinterpret_cast<char *>(&yLength), sizeof(yLength));
    inputFile.read(reinterpret_cast<char *>(&zLength), sizeof(zLength));
    inputFile.read(reinterpret_cast<char *>(&alpha), sizeof(alpha));
    inputFile.read(reinterpret_cast<char *>(&beta), sizeof(beta));
    inputFile.read(reinterpret_cast<char *>(&gamma), sizeof(gamma));
    inputFile.read(reinterpret_cast<char *>(&mapc), sizeof(mapc));
    inputFile.read(reinterpret_cast<char *>(&mapr), sizeof(mapr));
    inputFile.read(reinterpret_cast<char *>(&maps), sizeof(maps));
    inputFile.read(reinterpret_cast<char *>(&dMin), sizeof(dMin));
    inputFile.read(reinterpret_cast<char *>(&dMax), sizeof(dMax));
    inputFile.read(reinterpret_cast<char *>(&dMean), sizeof(dMean));
    inputFile.read(reinterpret_cast<char *>(&ispg), sizeof(ispg));
    inputFile.read(reinterpret_cast<char *>(&nsymbt), sizeof(nsymbt));
    inputFile.read(reinterpret_cast<char *>(extra), 25*sizeof(int));
    inputFile.read(reinterpret_cast<char *>(&xOrigin), sizeof(xOrigin));
    inputFile.read(reinterpret_cast<char *>(&yOrigin), sizeof(yOrigin));
    inputFile.read(reinterpret_cast<char *>(&zOrigin), sizeof(zOrigin));
    inputFile.read(reinterpret_cast<char *>(map), 4*sizeof(char));
    inputFile.read(reinterpret_cast<char *>(&machineStamp), sizeof(machineStamp));
    inputFile.read(reinterpret_cast<char *>(&rms), sizeof(rms));
    inputFile.read(reinterpret_cast<char *>(&nlabl), sizeof(nlabl));
    inputFile.read(reinterpret_cast<char *>(label), 10*80*sizeof(char));


    if (nx < 0 || ny < 0 || nz < 0) {
        return;
    }


    cube = new float **[nx];

    for(int i = 0; i < nx; i++)
    {
        cube[i] = new float *[ny];
        for(int j = 0; j < ny; j++)
            cube[i][j] = new float [nz];
    }

    for(int i = 0; i < nz; i++)
        for(int j = 0; j < ny; j++)
            for(int k = 0; k < nx; k++)
                inputFile.read(reinterpret_cast<char *>(&cube[k][j][i]), sizeof(float));

    inputFile.close();

    float min = 10000;
    float max = -10000;
    float sum = 0;

    for(int i = 0; i < nz; i++) {
        for(int j = 0; j < ny; j++) {
            for(int k = 0; k < nx; k++) {

                if ( cube[k][j][i] < min ) {
                    min = cube[k][j][i];
                }

                if ( cube[k][j][i] > max ) {
                    max = cube[k][j][i];
                }

                sum += cube[k][j][i];
            }
        }
    }
}



/*
Write MRC file
*/

void MRC::write(string file)
{
    fstream outputFile(file.c_str(), ios::out|ios::binary);
    if(!outputFile)
    {
       exit(1);
    }

    outputFile.write(reinterpret_cast<char *>(&nx), sizeof(nx));
    outputFile.write(reinterpret_cast<char *>(&ny), sizeof(ny));
    outputFile.write(reinterpret_cast<char *>(&nz), sizeof(nz));
    outputFile.write(reinterpret_cast<char *>(&mode), sizeof(mode));
    outputFile.write(reinterpret_cast<char *>(&nxStart), sizeof(nxStart));
    outputFile.write(reinterpret_cast<char *>(&nyStart), sizeof(nyStart));
    outputFile.write(reinterpret_cast<char *>(&nzStart), sizeof(nzStart));
    outputFile.write(reinterpret_cast<char *>(&mx), sizeof(mx));
    outputFile.write(reinterpret_cast<char *>(&my), sizeof(my));
    outputFile.write(reinterpret_cast<char *>(&mz), sizeof(mz));
    outputFile.write(reinterpret_cast<char *>(&xLength), sizeof(xLength));
    outputFile.write(reinterpret_cast<char *>(&yLength), sizeof(yLength));
    outputFile.write(reinterpret_cast<char *>(&zLength), sizeof(zLength));
    outputFile.write(reinterpret_cast<char *>(&alpha), sizeof(alpha));
    outputFile.write(reinterpret_cast<char *>(&beta), sizeof(beta));
    outputFile.write(reinterpret_cast<char *>(&gamma), sizeof(gamma));
    outputFile.write(reinterpret_cast<char *>(&mapc), sizeof(mapc));
    outputFile.write(reinterpret_cast<char *>(&mapr), sizeof(mapr));
    outputFile.write(reinterpret_cast<char *>(&maps), sizeof(maps));
    outputFile.write(reinterpret_cast<char *>(&dMin), sizeof(dMin));
    outputFile.write(reinterpret_cast<char *>(&dMax), sizeof(dMax));
    outputFile.write(reinterpret_cast<char *>(&dMean), sizeof(dMean));
    outputFile.write(reinterpret_cast<char *>(&ispg), sizeof(ispg));
    outputFile.write(reinterpret_cast<char *>(&nsymbt), sizeof(nsymbt));
    outputFile.write(reinterpret_cast<char *>(extra), 25*sizeof(int));
    outputFile.write(reinterpret_cast<char *>(&xOrigin), sizeof(xOrigin));
    outputFile.write(reinterpret_cast<char *>(&yOrigin), sizeof(yOrigin));
    outputFile.write(reinterpret_cast<char *>(&zOrigin), sizeof(zOrigin));
    outputFile.write(reinterpret_cast<char *>(map), 4*sizeof(char));
    outputFile.write(reinterpret_cast<char *>(&machineStamp), sizeof(machineStamp));
    outputFile.write(reinterpret_cast<char *>(&rms), sizeof(rms));
    outputFile.write(reinterpret_cast<char *>(&nlabl), sizeof(nlabl));
    outputFile.write(reinterpret_cast<char *>(label), 10*80*sizeof(char));

    for(int i = 0; i < nz; i++)
        for(int j = 0; j < ny; j++)
            for(int k = 0; k < nx; k++)
                outputFile.write(reinterpret_cast<char *>(&cube[k][j][i]), sizeof(float));

    outputFile.close();
}




void MRC::showMRCFileDescription()
{
}



void MRC::writeChoppedMRC(double xMinIndex, double xMaxIndex, double yMinIndex, double yMaxIndex, double zMinIndex, double zMaxIndex, double xMin, double yMin, double zMin, string file)
{

    fstream outputFile(file.c_str(), ios::out|ios::binary);
    if(!outputFile)
    {
       exit(1);
    }

    double shiftXDueToFraction = xMinIndex;
    double shiftYDueToFraction = yMinIndex;
    double shiftZDueToFraction = zMinIndex;


    xMinIndex = (int)xMinIndex;
    xMaxIndex = (int)xMaxIndex;
    yMinIndex = (int)yMinIndex;
    yMaxIndex = (int)yMaxIndex;
    zMinIndex = (int)zMinIndex;
    zMaxIndex = (int)zMaxIndex;

    shiftXDueToFraction -= xMinIndex;   // the shift is due to floating point calculation, the min (x,y,x) position from PDB is physical coordinate,
    shiftYDueToFraction -= yMinIndex;   // so  the, index in mrc does not converge exactly on that, since we take floor(index), we need to shift
    shiftZDueToFraction -= zMinIndex;   // that much left for X, bottom for y and so for the z.


    int offsetColX = 2;
    int offsetRowY = 2;
    int offsetSecZ = 2;


    double voxelSizeX = (double)xLength / nx;
    double voxelSizeY = (double)yLength / ny;
    double voxelSizeZ = (double)zLength / nz;


    nx = (xMaxIndex - xMinIndex + 1) + offsetColX;
    ny = (yMaxIndex - yMinIndex + 1) + offsetRowY;
    nz = (zMaxIndex - zMinIndex + 1) + offsetSecZ;




    xLength = nx * voxelSizeX;
    yLength = ny * voxelSizeY;
    zLength = nz * voxelSizeZ;

    nxStart = 0;
    nyStart = 0;
    nzStart = 0;

    xOrigin = xMin - voxelSizeX * (offsetColX / 2) - voxelSizeX * shiftXDueToFraction;
    yOrigin = yMin - voxelSizeY * (offsetRowY / 2) - voxelSizeY * shiftYDueToFraction;
    zOrigin = zMin - voxelSizeZ * (offsetSecZ / 2) - voxelSizeZ * shiftZDueToFraction;

    mx = nx;
    my = ny;
    mz = nz;



    printf("\n nx=%d  ny=%d nz= %d\n", nx, ny, nz);

    float ***choppedCube;

    choppedCube = new float **[nx];

    for(int i = 0; i < nx; i++)
    {
        choppedCube[i] = new float *[ny];
        for(int j = 0; j < ny; j++)
            choppedCube[i][j] = new float [nz];
    }

    int l = 0;
    int m = 0;
    int n = 0;

    // The 92 is a value for a particular protein, need to update for other protein

    for(int i = 92 + zMinIndex - (offsetSecZ / 2); i < 92 + zMaxIndex + 1 + (offsetSecZ / 2); i++) {

        m = 0;
        for(int j = 92 + yMinIndex - (offsetRowY / 2); j < 92 + yMaxIndex + 1 + (offsetRowY / 2); j++) {

            n = 0;
            for(int k = 92 + xMinIndex - (offsetColX / 2); k < 92 + xMaxIndex + 1 + (offsetColX / 2); k++) {

                choppedCube[n][m][l] = cube[k][j][i];
                n++;
            }
            m++;
        }
        l++;
    }


    outputFile.write(reinterpret_cast<char *>(&nx), sizeof(nx));
    outputFile.write(reinterpret_cast<char *>(&ny), sizeof(ny));
    outputFile.write(reinterpret_cast<char *>(&nz), sizeof(nz));
    outputFile.write(reinterpret_cast<char *>(&mode), sizeof(mode));
    outputFile.write(reinterpret_cast<char *>(&nxStart), sizeof(nxStart));
    outputFile.write(reinterpret_cast<char *>(&nyStart), sizeof(nyStart));
    outputFile.write(reinterpret_cast<char *>(&nzStart), sizeof(nzStart));
    outputFile.write(reinterpret_cast<char *>(&mx), sizeof(mx));
    outputFile.write(reinterpret_cast<char *>(&my), sizeof(my));
    outputFile.write(reinterpret_cast<char *>(&mz), sizeof(mz));
    outputFile.write(reinterpret_cast<char *>(&xLength), sizeof(xLength));
    outputFile.write(reinterpret_cast<char *>(&yLength), sizeof(yLength));
    outputFile.write(reinterpret_cast<char *>(&zLength), sizeof(zLength));
    outputFile.write(reinterpret_cast<char *>(&alpha), sizeof(alpha));
    outputFile.write(reinterpret_cast<char *>(&beta), sizeof(beta));
    outputFile.write(reinterpret_cast<char *>(&gamma), sizeof(gamma));
    outputFile.write(reinterpret_cast<char *>(&mapc), sizeof(mapc));
    outputFile.write(reinterpret_cast<char *>(&mapr), sizeof(mapr));
    outputFile.write(reinterpret_cast<char *>(&maps), sizeof(maps));
    outputFile.write(reinterpret_cast<char *>(&dMin), sizeof(dMin));
    outputFile.write(reinterpret_cast<char *>(&dMax), sizeof(dMax));
    outputFile.write(reinterpret_cast<char *>(&dMean), sizeof(dMean));
    outputFile.write(reinterpret_cast<char *>(&ispg), sizeof(ispg));
    outputFile.write(reinterpret_cast<char *>(&nsymbt), sizeof(nsymbt));
    outputFile.write(reinterpret_cast<char *>(extra), 25*sizeof(int));
    outputFile.write(reinterpret_cast<char *>(&xOrigin), sizeof(xOrigin));
    outputFile.write(reinterpret_cast<char *>(&yOrigin), sizeof(yOrigin));
    outputFile.write(reinterpret_cast<char *>(&zOrigin), sizeof(zOrigin));
    outputFile.write(reinterpret_cast<char *>(map), 4*sizeof(char));
    outputFile.write(reinterpret_cast<char *>(&machineStamp), sizeof(machineStamp));
    outputFile.write(reinterpret_cast<char *>(&rms), sizeof(rms));
    outputFile.write(reinterpret_cast<char *>(&nlabl), sizeof(nlabl));
    outputFile.write(reinterpret_cast<char *>(label), 10*80*sizeof(char));




    printf("\nl = %d , m = %d n = %d nz:%d ny:%d nx:%d", l,m,n,nz, ny, nx);


    for(int i = 0; i < nz; i++)
        for(int j = 0; j < ny; j++)
            for(int k = 0; k < nx; k++)
                outputFile.write(reinterpret_cast<char *>(&choppedCube[k][j][i]), sizeof(float));

    outputFile.close();

}



/*
This is the actual code that write MRC based on chain.

*/
void MRC::writeChoppedMRCBasedOnChain(double xMinIndex, double xMaxIndex, double yMinIndex, double yMaxIndex, double zMinIndex, double zMaxIndex, double xMin, double yMin, double zMin, string file, vector<Coordinate> atomicCoordinate)
{
    fstream outputFile(file.c_str(), ios::out|ios::binary);
    if(!outputFile)
    {
        exit(1);
    }

    xMinIndex = (int)xMinIndex;
    xMaxIndex = (int)xMaxIndex;
    yMinIndex = (int)yMinIndex;
    yMaxIndex = (int)yMaxIndex;
    zMinIndex = (int)zMinIndex;
    zMaxIndex = (int)zMaxIndex;

    double voxelSizeX = (double)xLength / nx;
    double voxelSizeY = (double)yLength / ny;
    double voxelSizeZ = (double)zLength / nz;

    int offsetColX = ((5.0 / voxelSizeX) < 4 ? 2: (5 / voxelSizeX) ); // 2; // 12;
    int offsetRowY = ((5.0 / voxelSizeY) < 4 ? 2: (5 / voxelSizeY));  //2; //12;
    int offsetSecZ = ((5.0 / voxelSizeZ) < 4 ? 2: (5 / voxelSizeZ) ); //2; //12;

    int nxChain = (xMaxIndex - xMinIndex + 1) + offsetColX;
    int nyChain = (yMaxIndex - yMinIndex + 1) + offsetRowY;
    int nzChain = (zMaxIndex - zMinIndex + 1) + offsetSecZ;

    float xLengthChain = nxChain * voxelSizeX;
    float yLengthChain = nyChain * voxelSizeY;
    float zLengthChain = nzChain * voxelSizeZ;

    int nxChainStart = 0;
    int nyChainStart = 0;
    int nzChainStart = 0;

    int mxChain = nxChain;
    int myChain = nyChain;
    int mzChain = nzChain;

    float ***choppedCube;
    choppedCube = new float **[nx];

    for(int i = 0; i < nx; i++)
    {
        choppedCube[i] = new float *[ny];
        for(int j = 0; j < ny; j++)
            choppedCube[i][j] = new float [nz];
    }

    float densityMinimum = 10000;
    float densityMaximum = - 10000;
    float densityTotal   = 0;
    int numOfVoxels = 0;


    int r = 0;
    for(int k =  zMinIndex - (offsetSecZ / 2); k <  zMaxIndex + 1 + (offsetSecZ / 2); k++) {

        int  q = 0;
        int c = k - nzStart;

        for(int j = yMinIndex - (offsetRowY / 2); j < yMaxIndex + 1 + (offsetRowY / 2); j++) {

            int p = 0;
            int b = j - nyStart;
            for(int i =  xMinIndex - (offsetColX / 2); i <  xMaxIndex + 1 + (offsetColX / 2); i++) {
                int a = i - nxStart;

                if (cube[a][b][c] > 0 ) {
                    choppedCube[p][q][r] = cube[a][b][c];
                } else {

                    choppedCube[p][q][r] = 0;

                }

                if(choppedCube[p][q][r] <  densityMinimum) {

                    densityMinimum = choppedCube[p][q][r];
                }

                if (choppedCube[p][q][r] >  densityMaximum) {

                    densityMaximum = choppedCube[p][q][r];
                }

                densityTotal += choppedCube[p][q][r];
                numOfVoxels ++;
                p++;
            }
            q++;
        }
        r++;
    }


    float xChainOrigin = xOrigin + ( xMinIndex - (offsetColX / 2)) * voxelSizeX;
    float yChainOrigin = yOrigin + ( yMinIndex - (offsetRowY / 2)) * voxelSizeY;
    float zChainOrigin = zOrigin + ( zMinIndex - (offsetSecZ / 2)) * voxelSizeZ;


    //Update the dMin, dMax, dMean, rms so that those are disregared by other software.
    //http://www.ccpem.ac.uk/mrc_format/mrc2014.php#note5

    dMin = densityMinimum;
    dMax = densityMaximum;
    dMean = (float) densityTotal / (nx * ny * nz);
    //rms = -1;

    outputFile.write(reinterpret_cast<char *>(&nxChain), sizeof(nxChain));
    outputFile.write(reinterpret_cast<char *>(&nyChain), sizeof(nyChain));
    outputFile.write(reinterpret_cast<char *>(&nzChain), sizeof(nzChain));
    outputFile.write(reinterpret_cast<char *>(&mode), sizeof(mode));
    outputFile.write(reinterpret_cast<char *>(&nxChainStart), sizeof(nxChainStart));
    outputFile.write(reinterpret_cast<char *>(&nyChainStart), sizeof(nyChainStart));
    outputFile.write(reinterpret_cast<char *>(&nzChainStart), sizeof(nzChainStart));
    outputFile.write(reinterpret_cast<char *>(&mxChain), sizeof(mxChain));
    outputFile.write(reinterpret_cast<char *>(&myChain), sizeof(myChain));
    outputFile.write(reinterpret_cast<char *>(&mzChain), sizeof(mzChain));
    outputFile.write(reinterpret_cast<char *>(&xLengthChain), sizeof(xLengthChain));
    outputFile.write(reinterpret_cast<char *>(&yLengthChain), sizeof(yLengthChain));
    outputFile.write(reinterpret_cast<char *>(&zLengthChain), sizeof(zLengthChain));
    outputFile.write(reinterpret_cast<char *>(&alpha), sizeof(alpha));
    outputFile.write(reinterpret_cast<char *>(&beta), sizeof(beta));
    outputFile.write(reinterpret_cast<char *>(&gamma), sizeof(gamma));
    outputFile.write(reinterpret_cast<char *>(&mapc), sizeof(mapc));
    outputFile.write(reinterpret_cast<char *>(&mapr), sizeof(mapr));
    outputFile.write(reinterpret_cast<char *>(&maps), sizeof(maps));
    outputFile.write(reinterpret_cast<char *>(&dMin), sizeof(dMin));
    outputFile.write(reinterpret_cast<char *>(&dMax), sizeof(dMax));
    outputFile.write(reinterpret_cast<char *>(&dMean), sizeof(dMean));
    outputFile.write(reinterpret_cast<char *>(&ispg), sizeof(ispg));
    outputFile.write(reinterpret_cast<char *>(&nsymbt), sizeof(nsymbt));
    outputFile.write(reinterpret_cast<char *>(extra), 25*sizeof(int));
    outputFile.write(reinterpret_cast<char *>(&xChainOrigin), sizeof(xChainOrigin));
    outputFile.write(reinterpret_cast<char *>(&yChainOrigin), sizeof(yChainOrigin));
    outputFile.write(reinterpret_cast<char *>(&zChainOrigin), sizeof(zChainOrigin));
    outputFile.write(reinterpret_cast<char *>(map), 4*sizeof(char));
    outputFile.write(reinterpret_cast<char *>(&machineStamp), sizeof(machineStamp));
    outputFile.write(reinterpret_cast<char *>(&rms), sizeof(rms));
    outputFile.write(reinterpret_cast<char *>(&nlabl), sizeof(nlabl));
    outputFile.write(reinterpret_cast<char *>(label), 10*80*sizeof(char));

    for(int i = 0; i < nzChain; i++)
        for(int j = 0; j < nyChain; j++)
            for(int k = 0; k < nxChain; k++)
                outputFile.write(reinterpret_cast<char *>(&choppedCube[k][j][i]), sizeof(float));

    outputFile.close();
}

void MRC::boundingCube(double xMin, double xMax, double yMin, double yMax, double zMin, double zMax, string choppedMRCPath)
{

    double voxelSizeX = (double)xLength / nx;
    double voxelSizeY = (double)yLength / ny;
    double voxelSizeZ = (double)zLength / nz;


    double xIndexMin = - (double)(xOrigin - xMin) / voxelSizeX;
    double xIndexMax =  (double)(xMax -  xOrigin) / voxelSizeX;
    double yIndexMin = - (double)(yOrigin - yMin) / voxelSizeY;
    double yIndexMax =  (double)(yMax -  yOrigin) / voxelSizeY;
    double zIndexMin = - (double)(zOrigin - zMin) / voxelSizeZ;
    double zIndexMax =  (double)(zMax -  zOrigin) / voxelSizeZ;
}


void MRC::boundingCubeBasedOnChain(double xMin, double xMax, double yMin, double yMax, double zMin, double zMax, string choppedMRCPath, vector<Coordinate> atomicCoordinate)
{

    double voxelSizeX = (double)xLength / nx;
    double voxelSizeY = (double)yLength / ny;
    double voxelSizeZ = (double)zLength / nz;

    double xIndexMin = - (double)(xOrigin - xMin) / voxelSizeX;
    double xIndexMax =  (double)(xMax -  xOrigin) / voxelSizeX;
    double yIndexMin = - (double)(yOrigin - yMin) / voxelSizeY;
    double yIndexMax =  (double)(yMax -  yOrigin) / voxelSizeY;
    double zIndexMin = - (double)(zOrigin - zMin) / voxelSizeZ;
    double zIndexMax =  (double)(zMax -  zOrigin) / voxelSizeZ;


    writeChoppedMRCBasedOnChain(xIndexMin, xIndexMax, yIndexMin, yIndexMax, zIndexMin, zIndexMax, xMin, yMin, zMin, choppedMRCPath, atomicCoordinate);
}


MRC::~MRC()
{

}
