#include <iostream>
#include <math.h>
//#include <sys/stat.h>
//#include <vector>
#include <sstream>
#include <fstream>

using namespace std;

int main()
{
    ofstream fout("testdl.txt");
    fout << "Start implement mrc..." << endl;

    fout << "test succesful" << endl;

    fout << "Have finished..." << endl;
    fout << flush;
    fout.close();
    return 0;
}
