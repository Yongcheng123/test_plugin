#include <iostream>
#include <math.h>
#include <sys/stat.h>
#include "MRC.h"
#include <vector>
#include <sstream>

using namespace std;

vector<string> split (const string &s, char delim) {
    vector<string> result;
    stringstream ss (s);
    string item;

    while (getline (ss, item, delim)) {
        result.push_back (item);
    }

    return result;
}

int main(int argc, const char **argv) 
{ 
    if(argc < 7)
    {
        cout << "ERROR: Not enough args" << endl;
        return 0;
    }

    string mrcFile = argv[1];
    string pdbFile = argv[2];
    string chain = argv[3];
    string predictionFilePath = argv[4];
    string outputDirectoryPath = argv[5];
    string mapsFolderPath = argv[6];

    string prefix = "_pred_";

    bool trueLabels = false;

    if(argc > 7)
    {
        prefix = "_true_";
        trueLabels = true;
    }

    outputDirectoryPath = outputDirectoryPath + "/" + mrcFile + "_" + pdbFile + "/";
    mkdir(outputDirectoryPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    string outputFileHelix = outputDirectoryPath + chain + prefix + "helix.mrc";
    string outputFileStrand = outputDirectoryPath + chain + prefix + "sheet.mrc";

    string mrcFilePath = mapsFolderPath + mrcFile + "_" + pdbFile + "/" + chain + ".mrc";

    MRC modifiedMap;
    modifiedMap.EMRead(mrcFilePath);

    //set all voxels to 0
    for(int i = 0; i < modifiedMap.getNx(); i++)
    {
        for(int j = 0; j < modifiedMap.getNy(); j++)
        {
            for(int k = 0; k < modifiedMap.getNz(); k++)
            {
               modifiedMap.cube[i][j][k] = 0;
            }
        }
    }

    //read predictions and adjust new map
    ifstream inFile;

    inFile.open(predictionFilePath.c_str());
    int xCoord = 0;
    int yCoord = 0;
    int zCoord = 0;
    int label = 0;
    int iter = 0;
    while(!inFile.eof())
    {
        if(!trueLabels)
        {
            inFile >> xCoord >> yCoord >> zCoord  >> label;
        }
        else
        {
            string temp;
            inFile >> temp;
            vector<string> temps = split(temp, ',');
            if(temps.size() != 5)
            {
                break;
            }
            xCoord = stoi(temps[0]);
            yCoord = stoi(temps[1]);
            zCoord = stoi(temps[2]);
            label = stoi(temps[4]);
        }
        if(xCoord < modifiedMap.getNx() && yCoord < modifiedMap.getNy() && zCoord < modifiedMap.getNz())
        {
            if(label == 1)
            {
                modifiedMap.cube[xCoord][yCoord][zCoord] = 1;
            }            
            else
            {
                modifiedMap.cube[xCoord][yCoord][zCoord] = 0;
            }
        }
    }
    inFile.close();
    modifiedMap.write(outputFileHelix);

    for(int i = 0; i < modifiedMap.getNx(); i++)
        for(int j = 0; j < modifiedMap.getNy(); j++)
            for(int k = 0; k < modifiedMap.getNz(); k++)
                modifiedMap.cube[i][j][k] = 0;

    //For Strands
    inFile.open(predictionFilePath.c_str());
    while(!inFile.eof())
    {
        if(!trueLabels)
        {
            inFile >> xCoord >> yCoord >> zCoord  >> label;
        }
        else
        {
            string temp;
            inFile >> temp;
            vector<string> temps = split(temp, ',');
            if(temps.size() != 5)
            {
                break;
            }
            xCoord = stoi(temps[0]);
            yCoord = stoi(temps[1]);
            zCoord = stoi(temps[2]);
            label = stoi(temps[4]);
        }
        if(xCoord < modifiedMap.getNx() && yCoord < modifiedMap.getNy() && zCoord < modifiedMap.getNz())
        {
            if(label == 2)
                modifiedMap.cube[xCoord][yCoord][zCoord] = 1;
            else
                modifiedMap.cube[xCoord][yCoord][zCoord] = 0;
        }
    }
    inFile.close();
    modifiedMap.write(outputFileStrand);

    return 0;
}
