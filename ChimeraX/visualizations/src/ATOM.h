#ifndef ____ATOM__
#define ____ATOM__

#include <stdio.h>
#include <string>

using namespace std;

struct Coordinate
{
    double xCor;
    double yCor;
    double zCor;

};

struct BoundingCoordinate
{
    Coordinate minPoints;
    Coordinate maxPoints;
};

class ATOM
{
private:
    Coordinate atomCoordinate;
    string chainName;

public:

    ATOM(){
        atomCoordinate.xCor = 0;
        atomCoordinate.yCor = 0;
        atomCoordinate.zCor = 0;
        chainName = "NULL";
    };

    void setValue (Coordinate coordinate, string chain);
    Coordinate getCoordinate();
    string getChainName();
    ~ATOM();
};

#endif /* defined(____ATOM__) */
