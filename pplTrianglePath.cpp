#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include<math.h>
// #include "ppl.hh"
// using namespace Parma_Polyhedra_Library;
// using namespace Parma_Polyhedra_Library::IO_Operators;
// using namespace std;


#include "pplTrianglePath.h"



int main(){

  

    for(int m=0; m<numOfExpression; m++){
       
       grpPolyhedron.add_constraint(grpCon[m]);
   }

    cout<<"\n grpPolyhedron.minimized_constraints(); before translation\n " ;
    cout<<grpPolyhedron.minimized_constraints()<<endl;

    // for computing previous cube
    NNC_Polyhedron originalCube(3);
    originalCube = grpPolyhedron;

  


    //For Collision Detection 
    NNC_Polyhedron pathHullPolyhedron(3);
    pathHullPolyhedron = grpPolyhedron;

    




    

    Linear_Expression e0 = 2*xp0+1;
    Linear_Expression e1 = zp0+1;
    Linear_Expression e3 = 1000*zp0+866;
    Linear_Expression e2 = 2*xp0-1;
    


    switch(dnnOutput){
        case 0 : 
                grpPolyhedron.affine_image(zp0, e3,1000);
                grpPolyhedron.affine_image(xp0, e0,2);
                break;

        case 1 : 

                grpPolyhedron.affine_image(zp0, e1);
                break;

        case 2 : 
                grpPolyhedron.affine_image(zp0, e3,1000);
                grpPolyhedron.affine_image(xp0, e2,2);
                break;

    }

    // cout<<"\nP[0].affine_image\n " ;
    // cout<<P[0].minimized_constraints()<<endl;

        cout<<"\n grpPolyhedron.minimized_constraints(); before hull, translated\n " ;
        cout<<grpPolyhedron.minimized_constraints()<<endl;



    //find and save pathHull constraints
    grpPolyhedron.poly_hull_assign(originalCube);
    cout<<"\n grpPolyhedron.poly_hull_assign(originalCube);\n " ;
    cout<<grpPolyhedron.minimized_constraints()<<endl;

    
    ofstream outFilePtr("triangleHullRegionpolyhedron.txt");
    outFilePtr <<  grpPolyhedron.minimized_constraints();
    outFilePtr.close();
    



    //Write previous cube constraints
   
    // Linear_Expression e0_0 = 2*xp0-1;
    // Linear_Expression e1_1 = zp0+1;
    // Linear_Expression e2_2 = 2*xp0+1;
    


    // switch(previousDNNOutput){
    //     case 0 : 
    //             originalCube.affine_image(zp0, e1_1);
    //             originalCube.affine_image(xp0, e2_2,2);
    //             break;

    //     case 1 : 

    //             originalCube.affine_image(zp0, e1_1);
    //             break;

    //     case 2 : 
    //             originalCube.affine_image(zp0, e1_1);
    //             originalCube.affine_image(xp0, e0_0,2);
    //             break;

    // }

    // cout<<"\noriginalCube.affine_image\n " ;
    // cout<<originalCube.minimized_constraints()<<endl;


    // ofstream outFilePtr2(previousCubeOutputFileName);
    // outFilePtr2 <<  originalCube.minimized_constraints();
    // outFilePtr2.close();
    // outFilePtr2.close();


    // for(int i = 0; i < numOfImages; ++i) {
    //     delete [] currImages[i];
    // }
    // delete [] currImages;
    // cout<<"pplCOllisionpath1.cpp finished"<<endl;
    return 1;
}