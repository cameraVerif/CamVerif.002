from pyparma import *
from z3 import *
import environment

vertices = environment.vertices
xp0 = Variable(0)
yp0 = Variable(1)
zp0 = Variable(2)



def computeRegionOfImages(currGroup, currGroupImages,currGroupDnnOutput,currGroupCubeCons):
    print("from computeRegionOfImages")
    pd  = NNC_Polyhedron(3)
    pd.add_constraints(environment.imageCons[currGroupImages[0]])
    
    print("Constraints before adding second image")
    print(pd.constraints())
    print("\n\n", pd.minimized_constraints())
    
    for i in range(1, len(currGroupImages)):
        print("\n\n current image :", currGroupImages[i])
        currImageCons = environment.imageCons[currGroupImages[i]]
        # print(currImageCons)
        
        # currImageConsList = currImageCons.split(",")
        # print(len(currImageConsList))
        # for j in range(0, len(currImageConsList)):
        pd2  = NNC_Polyhedron(3)
        pd2.add_constraints(environment.imageCons[currGroupImages[i]])
         
        
        # print(pd2.constraints())
        # print("\n\n", pd2.minimized_constraints())
        
        pd.poly_hull_assign(pd2)
        # print(pd.constraints())
        # print("\n\n", pd.minimized_constraints())
    
    # if(currGroup =="G_0" or currGroup =="G_1" or currGroup =="G_2"):    
    environment.groupRegionConsPPL[currGroup] = pd.minimized_constraints()
    
    print("pd constraints before intersection")
    print(pd.minimized_constraints())
    print("\n\n")

    pd3 = NNC_Polyhedron(3)
    print("currGroup : ",currGroup)
    pd3.add_constraints(currGroupCubeCons)
    # pd3.add_constraint(10*xp0>=1)
    # pd3.add_constraint(100*xp0<=11)
    # pd3.add_constraint(10*yp0>=45)
    # pd3.add_constraint(100*yp0<=451)
    # pd3.add_constraint(10*zp0>=1945)
    # pd3.add_constraint(100*zp0<=19451)
    
    # print("\n pd3 cons  before intersection ")
    # print(pd3.minimized_constraints())
    
    pd3.intersection_assign(pd)
    print("\n pd3 cons after intersection ")
    print(pd3.minimized_constraints())
    
    # pd3.affine_image()
    
        
    return pd3.minimized_constraints()
    
    