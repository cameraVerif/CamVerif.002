#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include<math.h>
// #include "ppl.hh"
// using namespace Parma_Polyhedra_Library;
// using namespace Parma_Polyhedra_Library::IO_Operators;
// using namespace std;


#include "pplcollisionPath1.h"
// #include "globalData_924_1.h"

// Variable xp0(0);
// Variable yp0(1);
// Variable zp0(2);


// int numOfImages, variableFlag;
// float currZP;
// string imagesDataFileName = "singleImageData.txt";
// string previousCubeOutputFileName = "pplpreviousCubeOutput.txt";
// string pathHullOutputFileName = "pplcollisionPath1_Output.txt";



// string groupFrustumConsFileName = "pplGroupConsInput.txt"; 

// // void generatePolyhedronConstraints(int currImages[][12], int i,NNC_Polyhedron *p){
// void generatePolyhedronConstraints(int **currImages, int i,NNC_Polyhedron *p){
//     	for(int j=0; j<numOfVertices; j++){
// 			int x = vertices[j*3+0];
// 			int y = vertices[j*3+1];
// 			int z = vertices[j*3+2];
//             // cout<<j<<" "<<x<<" "<<y<<" "<<z<<endl;
// 			// if( j == 0 || j==2|| j==4|| j==5|| j==6|| j==8|| j==10|| j==11){
//             if(z - currZP <0 ){
//                 p->add_constraint( ( (-67*(x -xp0) ) + (24*(z -zp0)*1 ) ) <= ((currImages[i][j*2+0])*(z -zp0)*1)  );
//                 p->add_constraint( ( (-67*(x -xp0) ) + (24*(z -zp0)*1 ) ) > (( (currImages[i][j*2+0]+1))*(z -zp0)*1)  );

//                 p->add_constraint( ( (67*(y -yp0) ) + (24*(z -zp0)*1 ) ) <= ((currImages[i][j*2+1])*(z -zp0)*1)  );
//                 p->add_constraint( ( (67*(y -yp0) ) + (24*(z -zp0)*1 ) ) > (( (currImages[i][j*2+1]+1))*(z -zp0)*1)  );
				
// 			}else  {
//                 //if(j == 1 || j==3 || j == 7)
//                  p->add_constraint( ( (-67*(x -xp0) ) + (24*(z -zp0)*1 ) ) >= ((currImages[i][j*2+0])*(z -zp0)*1)  );
//                 p->add_constraint( ( (-67*(x -xp0) ) + (24*(z -zp0)*1 ) ) < (( (currImages[i][j*2+0]+1))*(z -zp0)*1)  );

//                 p->add_constraint( ( (67*(y -yp0) ) + (24*(z -zp0)*1 ) ) >= ((currImages[i][j*2+1])*(z -zp0)*1)  );
//                 p->add_constraint( ( (67*(y -yp0) ) + (24*(z -zp0)*1 ) ) < (( (currImages[i][j*2+1]+1))*(z -zp0)*1)  );
			
// 			}
//             // else if( j==9){
//             //     //  p->add_constraint( ( (-39*(x -xp0)*10 ) + (245*(z -zp0)*1 ) )<= ((currImages[i][j*2+0]*10)*(z -zp0)*1)  );
//             //     // p->add_constraint( ( (-39*(x -xp0)*10 ) + (245*(z -zp0)*1 ) ) > (( (currImages[i][j*2+0]+1)*10)*(z -zp0)*1)  );

//             //     p->add_constraint( ( (39*(y -yp0)*10 ) + (245*(z -zp0)*1 ) ) >= ((currImages[i][j*2+1]*10)*(z -zp0)*1)  );
//             //     p->add_constraint( ( (39*(y -yp0)*10 ) + (245*(z -zp0)*1 ) ) < (( (currImages[i][j*2+1]+1)*10)*(z -zp0)*1)  );
				
//             // }
//         }
// }


int main(){
//     cout<<"\nPPLcollisionpath1.cpp"<<endl;
//     cout<<"PPLGenerating single cube constraints"<<endl;
//    std::fstream filePtr;
//    filePtr.open(imagesDataFileName,ios::in);

//    string currZString;
//    string numString;
//    string dnnOutputString;
//    string previousDnnOutputString;
    

// //    getline(filePtr, numString);
//    getline(filePtr, dnnOutputString);
//    getline(filePtr, previousDnnOutputString);
//    getline(filePtr, currZString);

//     int dnnOutput;
//     dnnOutput =  stoi(dnnOutputString);
//     // cout<<"dnn output = "<<dnnOutput<<endl;

//     int previousDNNOutput;
//     previousDNNOutput =  stoi(previousDnnOutputString);
//     // cout<<"previousDNNOutput = "<<previousDNNOutput<<endl;

// //    numOfImages =  stoi(numString);

//    currZP = stof(currZString);

//    std::cout<<"numofimages = "<<numOfImages<<std::endl;
//    variableFlag =  stoi(variableFlagString);
//    std::cout<<"variableFlag = "<<variableFlag<<std::endl;

    // numOfImages = 1;
    // int **currImages = new int*[numOfImages];
    // for(int i = 0; i < numOfImages; ++i) {
    //     currImages[i] = new int[numOfVertices*2];
    // }

    // string pixelString;
    // for(int i=0;i<numOfImages;i++){
    //     for(int j=0;j<numOfVertices*2;j++){

    //         getline(filePtr, pixelString);
    //         // std::cout<<pixelString<<std::endl;
    //         currImages[i][j] = stoi(pixelString);
    //     }
    // }

    // for(int i=0;i<numOfImages;i++){
    //     for(int j=0;j<numOfVertices*2;j++){
    //         printf("%d, ",currImages[i][j]);
    //     }
    //     printf("\n");
    // }

    // NNC_Polyhedron P[] = 
    //         {
    //         NNC_Polyhedron(3), NNC_Polyhedron(3), NNC_Polyhedron(3), NNC_Polyhedron(3), NNC_Polyhedron(3), NNC_Polyhedron(3),
       

    
    // };

    // // // numOfImages = 8;
    // // // NNC_Polyhedron *P = (NNC_Polyhedron*)malloc(sizeof(NNC_Polyhedron)*numOfImages);
    // // // for (int i = 0; i < numOfImages; i++) {
    // // //     P[i] = NNC_Polyhedron(3);
    // // // }

    // for(int k = 0; k <numOfImages;k++){
    //     // cout<<k<<std::endl;
    //     generatePolyhedronConstraints(currImages,k,&P[k]);
    //     // cout<<"\n\n " ;
    //     // cout<<P[k].constraints()<<endl;
    //     // cout<<P[k].minimized_constraints()<<endl;
    //     // cout<<"\n\n";
    //     // cout<<P[k].constraints()<<endl;
        
        
    //     if(k!=0){
    //         // cout<<"minimized_constraints"<<std::endl;
    //         P[0].poly_hull_assign(P[k]);
    //         // cout<<"\n\n " ;
    //         // cout<<P[0].minimized_constraints()<<endl;
    //     }
    // }


    // cout<<"\n\n " ;
    // cout<<P[0].minimized_constraints()<<endl;

   

    // char outputConstraints[10];
    // outputConstraints = P[0].minimized_constraints();
    // sprintf(outputConstraints," %s" ,P[0].minimized_constraints().print() ); 

    for(int m=0; m<numOfExpression; m++){
       
       grpPolyhedron.add_constraint(grpCon[m]);
   }

    // cout<<"\ngrpPolyhedron.minimized_constraints()\n " ;
    // cout<<grpPolyhedron.minimized_constraints()<<endl;

    // P[0].intersection_assign(grpPolyhedron);

    // for computing previous cube
    NNC_Polyhedron originalCube(3);
    originalCube = grpPolyhedron;

    // cout<<"\nP[0].intersection_assign(grpPolyhedron)\n " ;
    // cout<<P[0].minimized_constraints()<<endl;


    //For Collision Detection 
    NNC_Polyhedron pathHullPolyhedron(3);
    pathHullPolyhedron = grpPolyhedron;

    




    

    Linear_Expression e0 = 2*xp0-1;
    Linear_Expression e1 = zp0-1;
    Linear_Expression e3 = 1000*zp0-866;
    Linear_Expression e2 = 2*xp0+1;
    


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





    //find and save pathHull constraints
    // pathHullPolyhedron.poly_hull_assign(P[0]);
    // cout<<"\npathHullPolyhedron.poly_hull_assign(P[0]);\n " ;
    // cout<<pathHullPolyhedron.minimized_constraints()<<endl;

    
    ofstream outFilePtr("postRegionpolyhedron.txt");
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