#ifndef MRC_H
#define MRC_H
#include <iostream>
#include <fstream>
#include <vector>
#include  "ATOM.h"

using namespace std;


class MRC
{
private: int nx;//number of columns (fastest changing in map)
	 	 int ny;//number of rows
	 	 int nz;//number of sections (slowest changing in map)
	 	 int mode;//data type
    //0 image: signed 8-bit bytes range -128 to 127
    //1 image: 16-bit halfwords
    //2 image: 32-bit reals
    //3 transform: complex 16-bit integers
    //4 transform: complex 32-bit reals
    //6 image: unsigned 16-bit range 0 to 65535
         int nxStart;//number of first column in map (Defalut = 0)
	 	 int nyStart;//number of first row in map
	 	 int nzStart;//number of first section in map
	 	 int mx;//number of intervals along X
	 	 int my;//number of intervals along Y
	 	 int mz;//number of intervals along Z
	 	 float xLength;//cell dimensions in X (angstrom)
	 	 float yLength;//cell dimensions in Y (angstrom)
	 	 float zLength;//cell dimensions in Z (angstrom)
	 	 float alpha;//cell angle between Y and Z
	 	 float beta;//cell angle between X and Z
	 	 float gamma;//cell angle between X and Y
	 	 int mapc;//axis corresp to cols(1,2,3 for X,Y,Z)
	 	 int mapr;//axis corresp to rows(1,2,3 for X,Y,Z)
	 	 int maps;//axis corresp to sections(1,2,3 for X,Y,Z)
	 	 float dMin;//minimum density value
	 	 float dMax;//maximum density value
	 	 float dMean;//mean density value
	 	 int ispg;//space group number 0 or 1 (default = 0)
	 	 int nsymbt;//number of bytes used for symmetry data (0 or 80)
	 	 int extra[25];//extra space used for anything - 0 by default
	 	 float xOrigin;//origin in X used for transforms
	 	 float yOrigin;//origin in Y used for transforms
	 	 float zOrigin;//origin in Z used for transforms
	 	 char map[4];//character string 'MPA' to identify file type
	 	 int machineStamp;//machine stamp
	 	 float rms;//rms deviation of map from mean density
	 	 int nlabl;//number of labels being used
	 	 char label[10][80];//10 80-character text labels

public:

    float ***cube;
   // vector<AminoAcid> aminoAcids;

    MRC()//consturctor
    {
    }

    void surface(double cutOff);//get surface of the density map
    bool isSurface(int i, int j, int k, double cutOff);//check if point is on the surface

    void EMRead(std::string file);//read density map by EMan formate
    void write(string file);


    void showMRCFileDescription();
    void boundingCube(double xMin, double xMax, double yMin, double yMax, double zMin, double zMax, string choppedMRCPath);
    void boundingCubeBasedOnChain(double xMin, double xMax, double yMin, double yMax, double zMin, double zMax, string choppedMRCPath, vector<Coordinate> atomicCoordinate);


    void writeChoppedMRC(double xMinIndex, double xMaxIndex, double yMinIndex, double yMaxIndex, double zMinIndex, double zMaxIndex, double xMin, double yMin, double zMin, string file);

    void writeChoppedMRCBasedOnChain(double xMinIndex, double xMaxIndex, double yMinIndex, double yMaxIndex, double zMinIndex, double zMaxIndex, double xMin, double yMin, double zMin, string file, vector<Coordinate> atomicCoordinate);


    int getNx() const {return nx;}
    int getNy() const {return ny;}
    int getNz() const {return nz;}
    int getXLength() const {return xLength;}
    int getYLength() const {return yLength;}
    int getZLength() const {return zLength;}
    double getApixX() const {return xLength/mx;}
    double getApixY() const {return yLength/my;}
    double getApixZ() const {return zLength/mz;}

    ~MRC();
};


#endif
