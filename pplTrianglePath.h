#include "ppl.hh" 
using namespace Parma_Polyhedra_Library; 
using namespace Parma_Polyhedra_Library::IO_Operators;
using namespace std;
Variable xp0(0);
Variable yp0(1);
Variable zp0(2);
NNC_Polyhedron grpPolyhedron(3);
const int numOfExpression =5;


Constraint grpCon[numOfExpression] = { zp0-192==0, -100*yp0+451>=0, -20*xp0-39>=0, 1000*xp0+1951>=0, 2*yp0-9>=0 ,};

int dnnOutput =1;

