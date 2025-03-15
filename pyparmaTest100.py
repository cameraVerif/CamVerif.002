from pyparma import *
from z3 import *

import environment
import os


xp0 = Variable(0)
yp0 = Variable(1)
zp0 = Variable(2)


vertices = environment.vertices
nvertices = environment.nvertices


# current region cons : Constraint_System {-100*xp0+361>=0, -100*yp0+451>=0, -125*zp0+21181>=0, 500*zp0-84719>=0, 2*yp0-9>=0, 5*xp0-18>=0}
# next region cons :  Constraint_System {-100*xp0+411>=0, -100*yp0+451>=0, -500*zp0+84291>=0, 250*zp0-42143>=0, 2*yp0-9>=0, 10*xp0-41>=0}
# path hull cons  Constraint_System {-100*yp0+451>=0, -43300*xp0-25000*zp0+4392513>=0, -100*xp0+411>=0, 4330*xp0+2500*zp0-439183>=0, -125*zp0+21181>=0, 250*zp0-42143>=0, 5*xp0-18>=0, 2*yp0-9>=0}


pd4 = NNC_Polyhedron(3)
pd4.add_constraint(-100*xp0+361>=0)
pd4.add_constraint(-100*yp0+451>=0)
pd4.add_constraint(-125*zp0+21181>=0)
pd4.add_constraint(500*zp0-84719>=0)
pd4.add_constraint(2*yp0-9>=0)
pd4.add_constraint(5*xp0-18>=0)


print("current region cons : ",pd4.minimized_constraints())

currRegion = NNC_Polyhedron(3)
currRegion.add_constraint(-100*xp0+361>=0)
currRegion.add_constraint(-100*yp0+451>=0)
currRegion.add_constraint(-125*zp0+21181>=0)
currRegion.add_constraint(500*zp0-84719>=0)
currRegion.add_constraint(2*yp0-9>=0)
currRegion.add_constraint(5*xp0-18>=0)


pd5 = NNC_Polyhedron(3)
pd5.add_constraint(-100*xp0+411>=0)
pd5.add_constraint(-100*yp0+451>=0)
pd5.add_constraint(-500*zp0+84291>=0)
pd5.add_constraint(250*zp0-42143>=0)
pd5.add_constraint(2*yp0-9>=0)
pd5.add_constraint(10*xp0-41>=0)

print("next region cons : ",pd5.minimized_constraints())


triangle = 3202

x0 = vertices[nvertices[triangle*3+0]*3+0]  
y0 = vertices[nvertices[triangle*3+0]*3+1] 
z0 = vertices[nvertices[triangle*3+0]*3+2] 

x1 = vertices[nvertices[triangle*3+1]*3+0] 
y1 = vertices[nvertices[triangle*3+1]*3+1]
z1 = vertices[nvertices[triangle*3+1]*3+2]

x2 = vertices[nvertices[triangle*3+2]*3+0]
y2 = vertices[nvertices[triangle*3+2]*3+1]
z2 = vertices[nvertices[triangle*3+2]*3+2]


print("x0 = ",x0, " y0 = ",y0, " z0 = ",z0)
print("x1 = ",x1, " y1 = ",y1, " z1 = ",z1)
print("x2 = ",x2, " y2 = ",y2, " z2 = ",z2)

x0 = int(x0*pow(10,3)) 
y0 = int(y0*pow(10,3)) 
z0 = int(z0*pow(10,3)) 

x1 = int(x1*pow(10,3)) 
y1 = int(y1*pow(10,3)) 
z1 = int(z1*pow(10,3)) 

x2 = int(x2*pow(10,3)) 
y2 = int(y2*pow(10,3)) 
z2 = int(z2*pow(10,3)) 


print("x0 = ",x0, " y0 = ",y0, " z0 = ",z0)
print("x1 = ",x1, " y1 = ",y1, " z1 = ",z1)
print("x2 = ",x2, " y2 = ",y2, " z2 = ",z2)

trianglePolyhedron = NNC_Polyhedron(3,'empty')
trianglePolyhedron.add_generator(point( x0*xp0+y0*yp0+z0*zp0, pow(10,3) ))
trianglePolyhedron.add_generator(point( x1*xp0+y1*yp0+z1*zp0, pow(10,3) ))
trianglePolyhedron.add_generator(point( x2*xp0+y2*yp0+z2*zp0, pow(10,3) ))


print("trianglePolyhedron cons : ",trianglePolyhedron.minimized_constraints())


pd4.poly_hull_assign(pd5)

print("path hull cons ",pd4.minimized_constraints())

pd4.intersection_assign(trianglePolyhedron)

print("intersection cons ",pd4.minimized_constraints())



testPolyHe = NNC_Polyhedron(3,'empty')
testPolyHe.add_generator(point(1*xp0+1*yp0+1*zp0))
testPolyHe.add_generator(point(10*xp0+1*yp0+1*zp0))
testPolyHe.add_generator(point(1*xp0+10*yp0+1*zp0))
testPolyHe.add_generator(point(10*xp0+10*yp0+1*zp0))

# testPolyHe.add_generator(point(1*xp0+1*yp0+10*zp0))
# testPolyHe.add_generator(point(10*xp0+1*yp0+10*zp0))
# testPolyHe.add_generator(point(1*xp0+10*yp0+10*zp0))
# testPolyHe.add_generator(point(10*xp0+10*yp0+10*zp0))


print("testPolyHe cons : ",testPolyHe.minimized_constraints())


currIntersectionRegionConsString = str(pd4.minimized_constraints())
    
currIntersectionRegionConsString = currIntersectionRegionConsString.replace("x0","xp0")
currIntersectionRegionConsString = currIntersectionRegionConsString.replace("x1","yp0")
currIntersectionRegionConsString = currIntersectionRegionConsString.replace("x2","zp0")
currIntersectionRegionConsString = currIntersectionRegionConsString.replace("Constraint_System {"," ")
currIntersectionRegionConsString = currIntersectionRegionConsString.replace("}"," ")

currIntersectionRegionConsList = currIntersectionRegionConsString.split(",")

print(currIntersectionRegionConsList)

dnnOutput = 2

# e0 = 2*xp0+1
# e1 = zp0+1
# e3 = 1000*zp0+866
# e2 = 2*xp0-1

headerFilePre = "#include \"ppl.hh\" \nusing namespace Parma_Polyhedra_Library; \nusing namespace Parma_Polyhedra_Library::IO_Operators;\nusing namespace std;\nVariable xp0(0);\nVariable yp0(1);\nVariable zp0(2);\nNNC_Polyhedron grpPolyhedron(3);";

# #writes gropu frustum constraints 
pplGrpConsInputFile = open("pplTrianglePath.h",'w')
pplGrpConsInputFile.write(headerFilePre);
pplGrpConsInputFile.write("\nconst int numOfExpression ="+str(len(currIntersectionRegionConsList))+";\n\n");

pplGrpConsInputFile.write("\nConstraint grpCon[numOfExpression] = {");  
for k in range(0, len(currIntersectionRegionConsList)):
    pplGrpConsInputFile.write(str(currIntersectionRegionConsList[k])+",")
pplGrpConsInputFile.write("};\n\n"); 

pplGrpConsInputFile.write("int dnnOutput ="+str(dnnOutput) +";\n\n"); 
pplGrpConsInputFile.close()

tempstring = "touch pplTrianglePath.cpp"
print("touching file")
os.system(tempstring)

tempstring = "gcc pplTrianglePath.cpp -o pplTrianglePath -L/home2/habeebp/opt/include/ -L/home2/habeebp/opt/lib/ -I/home2/habeebp/opt/include/ -lstdc++ -lppl -lgmpxx -lgmp"
print("compiling pplTrianglePath.cpp")
os.system(tempstring)

tempstring = "./pplTrianglePath"
os.system(tempstring)

pplOutputFilePtr = open("triangleHullRegionpolyhedron.txt",'r')
print("\n\n From ppl\n")
preRegionpolyhedronConString = pplOutputFilePtr.read()
print(preRegionpolyhedronConString)
pplOutputFilePtr.close()

preRegionpolyhedronConString = str(preRegionpolyhedronConString)
preRegionpolyhedronConString = preRegionpolyhedronConString.replace("A","xp0")
preRegionpolyhedronConString = preRegionpolyhedronConString.replace("B","yp0")
preRegionpolyhedronConString = preRegionpolyhedronConString.replace("C","zp0")
preRegionpolyhedronConString = preRegionpolyhedronConString.replace(" = ","==")
preRegionpolyhedronConStringList = preRegionpolyhedronConString.split(",")


print(preRegionpolyhedronConStringList)

xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
newCons = And(True)
for r in range(0,len(preRegionpolyhedronConStringList)):
    newCons = simplify(And(newCons,eval(str(preRegionpolyhedronConStringList[r]))))


print(newCons)


currRegionPPL =  currRegion.minimized_constraints()

currImageSetConString = str(currRegionPPL)
currImageSetConString = currImageSetConString.replace("x0","xp0")
currImageSetConString = currImageSetConString.replace("x1","yp0")
currImageSetConString = currImageSetConString.replace("x2","zp0")
currImageSetConString = currImageSetConString.replace(" = ","==")
currImageSetConString = currImageSetConString.replace("Constraint_System {"," ")
currImageSetConString = currImageSetConString.replace("}"," ")
currImageSetConString = "And("+currImageSetConString+" )"
currGroupCons = eval(currImageSetConString)
print("Current image cons = ", currGroupCons)

currImageConsZ3 = currGroupCons   




intersectionRegionConZ3 = And(currImageConsZ3,newCons)

print("Final Intersection region to back track =")
print(simplify(intersectionRegionConZ3))
print("\n\nsimplified formula : ", simplify(intersectionRegionConZ3))




xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
xp1,yp1,zp1 = Reals('xp1 yp1 zp1')
# currentRegionPath =  currGroup
# currentRegionCons = intersectionRegionConZ3
# currRegionOutputToCheck = dnnOutput


ss100 = Solver()
ss100.add(simplify(intersectionRegionConZ3))
print("number of cons = ", len(ss100.assertions()))
print(ss100.check())
exit()



# simplified formula :  And(xp0 <= 361/100,
#     yp0 <= 451/100,
#     zp0 <= 21181/125,
#     zp0 >= 84719/500,
#     yp0 >= 9/2,
#     xp0 >= 18/5,
#     -1026672816869457792*xp0 +
#     7032846602368519296*yp0 +
#     6797686091442502656*zp0 >=
#     6651519140830031551,
#     -210936372094507788222603256140153756*xp0 +
#     102937657347787347646596845930545152*yp0 +
#     -121787743703526436618131210242583000*zp0 >=
#     -288100790352214570794622342260221807,
#     128334102108682224000*xp0 +
#     -879105825296064912000*yp0 +
#     -849710761430312832000*zp0 >=
#     -1631456463056745968387,
#     105468186047253894111301628070076878*xp0 +
#     -51468828673893673823298422965272576*yp0 +
#     60893871851763218309065605121291500*zp0 >=
#     132957045860626414718346535013786821)
# number of cons =  1


# simplified formula :  And(xp0 <= 361/100,
#     yp0 <= 451/100,
#     zp0 <= 21181/125,
#     zp0 >= 84719/500,
#     yp0 >= 9/2,
#     xp0 >= 18/5,
#     4330*xp0 + 2500*zp0 >= 439183,
#     -811829500*xp0 + 179302000*yp0 + 1701252000*zp0 >=
#     285605831481,
#     -43300*xp0 + -25000*zp0 >= -4392513,
#     811829500*xp0 + -179302000*yp0 + -1701252000*zp0 >=
#     -287485030463)
