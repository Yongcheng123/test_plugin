#include "ATOM.h"
#include <iostream>
using namespace std;

void ATOM:: setValue (Coordinate coordinate, string chain)
{
    atomCoordinate.xCor = coordinate.xCor;
    atomCoordinate.yCor = coordinate.yCor;
    atomCoordinate.zCor = coordinate.zCor;
    chainName = chain;
}

Coordinate ATOM:: getCoordinate()
{
    return atomCoordinate;
}

string ATOM:: getChainName()
{
    return chainName;
}

ATOM:: ~ATOM() {

};

