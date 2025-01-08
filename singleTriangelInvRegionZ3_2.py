from z3 import *
import environment
from time import sleep

vertices = environment.vertices
r = environment.r



def symInsideVertexPixelCos():
    pass



def symIntersectionPoint(plane,insideVertex, outsideVertex,xpixel,ypixel,xp0,yp0,zp0): 
    
    
    
   
    
    
    x0 = 0
    y0 = 0
    z0 = 0

    x1 = 0
    y1 = 0
    z1 = 0 
    
    x2 = 0
    y2 = 0
    z2 = 0 
    
    x3 = 0
    y3 = 0
    z3 = 0 
    
    
    if(plane == 0):
        # print("checking intersection with the top plane")
        dummyValue = -1000
        # plane0_v0 = [-canvasWidth*25.4/(2*focalLength),canvasHeight*25.4*1.34/(2*focalLength),focalLength]
        # plane0_v1 = [0,0,0]
        # plane0_v2 = [canvasWidth*25.4/(2*focalLength),canvasHeight*25.4*1.34/(2*focalLength),focalLength]    

        plane0_v0 = [-0.35820895522388063,0.35820895522388063,-1]
        plane0_v1 = [0.35820895522388063,0.35820895522388063,-1]
        plane0_v2 = [-358.20895522388063,358.20895522388063,-1000]    
        plane0_v3 = [358.20895522388063,358.20895522388063,-1000] 
        
        # print("plane coordinates :",plane0_v0,plane0_v1,plane0_v2)
    
        
        
        x0 = plane0_v0[0]
        y0 = plane0_v0[1]
        z0 = plane0_v0[2] 

        x1 = plane0_v1[0]
        y1 = plane0_v1[1]
        z1 = plane0_v1[2] 
        
        x2 = plane0_v2[0]
        y2 = plane0_v2[1]
        z2 = plane0_v2[2] 
        
        x3 = plane0_v3[0]
        y3 = plane0_v3[1]
        z3 = plane0_v3[2] 


        
    
    if(plane == 1):
        print("intersection with the bottom plane")
        
        

        plane0_v0 = [-0.35820895522388063,-0.35820895522388063,-1]
        plane0_v1 = [0.35820895522388063,-0.35820895522388063,-1]
        plane0_v2 = [-358.20895522388063,-358.20895522388063,-1000]    
        plane0_v3 = [358.20895522388063,-358.20895522388063,-1000] 
        
        # print("plane coordinates :",plane0_v0,plane0_v1,plane0_v2)
    
        # xk, yk, zk = Reals('xk yk zk')
        # u, v, w, g  = Reals('u v w g')
        # s1.check()
        # s1.add(u+v+w+g == 1)
        # s1.add(And(u>=0, v>=0, w>=0,g>=0))
        # s1.check()
        
        x0 = plane0_v0[0]
        y0 = plane0_v0[1]
        z0 = plane0_v0[2] 

        x1 = plane0_v1[0]
        y1 = plane0_v1[1]
        z1 = plane0_v1[2] 
        
        x2 = plane0_v2[0]
        y2 = plane0_v2[1]
        z2 = plane0_v2[2] 
        
        x3 = plane0_v3[0]
        y3 = plane0_v3[1]
        z3 = plane0_v3[2] 


      
    
    if(plane == 2):
            # print("checking intersection with the right plane")
           

            plane0_v0 = [0.35820895522388063,0.35820895522388063,-1]
            plane0_v1 = [0.35820895522388063,-0.35820895522388063,-1]
            plane0_v2 = [358.20895522388063,358.20895522388063,-1000]    
            plane0_v3 = [358.20895522388063,-358.20895522388063,-1000] 
            
            # print("plane coordinates :",plane0_v0,plane0_v1,plane0_v2)
        
            # xk, yk, zk = Reals('xk yk zk')
            # u, v, w, g  = Reals('u v w g')
            # s1.check()
            # s1.add(u+v+w+g == 1)
            # s1.add(And(u>=0, v>=0, w>=0,g>=0))
            # s1.check()
            
            x0 = plane0_v0[0]
            y0 = plane0_v0[1]
            z0 = plane0_v0[2] 

            x1 = plane0_v1[0]
            y1 = plane0_v1[1]
            z1 = plane0_v1[2] 
            
            x2 = plane0_v2[0]
            y2 = plane0_v2[1]
            z2 = plane0_v2[2] 
            
            x3 = plane0_v3[0]
            y3 = plane0_v3[1]
            z3 = plane0_v3[2] 


           
    
    if(plane == 3):
        # print("checking intersection with the left plane")
           

        plane0_v0 = [-0.35820895522388063,0.35820895522388063,-1]
        plane0_v1 = [-0.35820895522388063,-0.35820895522388063,-1]
        plane0_v2 = [-358.20895522388063,358.20895522388063,-1000]    
        plane0_v3 = [-358.20895522388063,-358.20895522388063,-1000] 
        
        # print("plane coordinates :",plane0_v0,plane0_v1,plane0_v2)
    
        # xk, yk, zk = Reals('xk yk zk')
        # u, v, w, g  = Reals('u v w g')
        # s1.check()
        # s1.add(u+v+w+g == 1)
        # s1.add(And(u>=0, v>=0, w>=0,g>=0))
        # s1.check()
        
        x0 = plane0_v0[0]
        y0 = plane0_v0[1]
        z0 = plane0_v0[2] 

        x1 = plane0_v1[0]
        y1 = plane0_v1[1]
        z1 = plane0_v1[2] 
        
        x2 = plane0_v2[0]
        y2 = plane0_v2[1]
        z2 = plane0_v2[2] 
        
        x3 = plane0_v3[0]
        y3 = plane0_v3[1]
        z3 = plane0_v3[2] 


                         
    
    if(plane ==5):
        # print("checking intersection with the near plane")
           

        plane0_v0 = [-0.35820895522388063,0.35820895522388063,-1]
        plane0_v1 = [-0.35820895522388063,-0.35820895522388063,-1]
        plane0_v2 = [0.35820895522388063,-0.35820895522388063,-1]    
        plane0_v3 = [0.35820895522388063,0.35820895522388063,-1] 
        
        # print("plane coordinates :",plane0_v0,plane0_v1,plane0_v2)
    
        # xk, yk, zk = Reals('xk yk zk')
        # u, v, w, g  = Reals('u v w g')
        # s1.check()
        # s1.add(u+v+w+g == 1)
        # s1.add(And(u>=0, v>=0, w>=0,g>=0))
        # s1.check()
        
        x0 = plane0_v0[0]
        y0 = plane0_v0[1]
        z0 = plane0_v0[2] 

        x1 = plane0_v1[0]
        y1 = plane0_v1[1]
        z1 = plane0_v1[2] 
        
        x2 = plane0_v2[0]
        y2 = plane0_v2[1]
        z2 = plane0_v2[2] 
        
        x3 = plane0_v3[0]
        y3 = plane0_v3[1]
        z3 = plane0_v3[2] 



    p0,q0 = Reals('p0 q0')
    u0, v0, w0, g0 = Reals('u0 v0 w0 g0')
    
    # ypixel = 0
    
    f1 = p0+q0 == 1
    f2 = And(p0>=0,q0>=0)
    f3 = u0+v0+w0+g0 == 1
    f4 = And(u0>=0, v0>=0, w0>=0,g0>=0)
    
    
    f5 = (p0* (vertices[insideVertex*3+0] - xp0) + (q0)* (vertices[outsideVertex*3+0] - xp0) )  == (u0*x0+v0*x1+w0*x2+g0*x3)
    f6 = (p0* (vertices[insideVertex*3+1] - yp0) + (q0)* (vertices[outsideVertex*3+1] - yp0) )  == (u0*y0+v0*y1+w0*y2+g0*y3)
    f7 = (p0* (vertices[insideVertex*3+2] - zp0) + (q0)* (vertices[outsideVertex*3+2] - zp0) )  == (u0*z0+v0*z1+w0*z2+g0*z3)
    
    
    # cons1 = ( (((-67*(  (p0* (vertices[insideVertex*3+0] - xp0) + (1-p0)* (vertices[outsideVertex*3+0] - xp0) )  ) )/ ( (p0* (vertices[insideVertex*3+2] - zp0) + (1-p0)* (vertices[outsideVertex*3+2] - zp0) ) ) )+ 24 ) >= xpixel )
    # cons2 = ( (((-67*(  (p0* (vertices[insideVertex*3+0] - xp0) + (1-p0)* (vertices[outsideVertex*3+0] - xp0) )  ) )/ ( (p0* (vertices[insideVertex*3+2] - zp0) + (1-p0)* (vertices[outsideVertex*3+2] - zp0) ) ) )+ 24 ) <  xpixel + 1 )
    # cons3 = ( ((( 67*(  (p0* (vertices[insideVertex*3+1] - yp0) + (1-p0)* (vertices[outsideVertex*3+1] - yp0) )  ) )/ ( (p0* (vertices[insideVertex*3+2] - zp0) + (1-p0)* (vertices[outsideVertex*3+2] - zp0) ) ) )+ 24 ) >=  ypixel )
    # cons4 = ( ((( 67*(  (p0* (vertices[insideVertex*3+1] - yp0) + (1-p0)* (vertices[outsideVertex*3+1] - yp0) )  ) )/ ( (p0* (vertices[insideVertex*3+2] - zp0) + (1-p0)* (vertices[outsideVertex*3+2] - zp0) ) ) )+ 24 ) <  ypixel + 1 )
    
    cons1 = ( (((-67*(  (u0*x0+v0*x1+w0*x2+g0*x3)  ) )/ ( (u0*z0+v0*z1+w0*z2+g0*z3) ) )+ 24 ) >= xpixel )
    cons2 = ( (((-67*(  (u0*x0+v0*x1+w0*x2+g0*x3)  ) )/ ( (u0*z0+v0*z1+w0*z2+g0*z3) ) )+ 24 ) <  xpixel + 1 )
    cons3 = ( ((( 67*(  (u0*y0+v0*y1+w0*y2+g0*y3)  ) )/ ( (u0*z0+v0*z1+w0*z2+g0*z3) ) )+ 24 ) >=  ypixel )
    cons4 = ( ((( 67*(  (u0*y0+v0*y1+w0*y2+g0*y3)  ) )/ ( (u0*z0+v0*z1+w0*z2+g0*z3) ) )+ 24 ) <  ypixel + 1 )
    
    
    # formula = Exists((p0,q0,u0,v0,w0,g0), And(f1,f2,f3,f4,f5,f6,f7,cons1,cons2,cons3,cons4))
    
    formula = Exists((p0,q0,u0,v0,w0,g0), And(f1,f2,f3,f4,f5,f6,f7,cons1,cons2,cons3,cons4))
    
    # print("eleminating")
    # print(formula)
    # g  = Goal()
    # g.add(formula)
    # t3 = Tactic('qe')
    # # t5 = Tactic('simplify')
    # # t  = Then( t3,t5)
    # t = t3
    # # print("\n\n\nAfter elimination")
    # resultFormula = t(g)
    
    # print("elemination done")
    
    # tempExpression = And(True)
    # for i in range(0,len(resultFormula[0])):
    #     tempExpression = simplify(And(tempExpression,resultFormula[0][i]))
    
    # print(tempExpression)
    # return simplify(tempExpression)

    return formula
    
    
    
     
    
    
    
    
    
    
    
    
    


        

def computeRegionConstraintZ3(currGroupName,currZP,numberOfFullyInsideVertices, insideVertexDetailsToPPL, numberOfIntersectingEdges,\
    intersectingEdgeDataToPPL,posXp1,posYp1,posZp1,mxp,myp,mzp,outcodeP0,currImageName,xp0,yp0,zp0):
    
    pixelMapConstraint = [And(True)]*(numberOfFullyInsideVertices+numberOfIntersectingEdges)
    
    # finalConstraintToReturn = And(True)
    print("numberOfFullyInsideVertices = ",numberOfFullyInsideVertices)
    for i in range(0,numberOfFullyInsideVertices): 
        print("\n\nfully inside vertex ==> ", i)       
        currentVertexIndex = insideVertexDetailsToPPL[i][0]
        xpixel = insideVertexDetailsToPPL[i][1]
        ypixel = insideVertexDetailsToPPL[i][2]
        
        x = vertices[currentVertexIndex*3+0];
        y = vertices[currentVertexIndex*3+1];
        z = vertices[currentVertexIndex*3+2];
    
    
        VertexPixelConstraint1 = And( 
                        ( (((-67*(x -xp0) )/ (z -zp0) )+ 24 )>= xpixel ),\
                        ( (((-67*(x -xp0) )/ (z -zp0) )+ 24 )< xpixel+1),\
                        ( (((67*(y -yp0)  )/ (z -zp0) )+ 24 ) >= ypixel ),\
                        ( (((67*(y -yp0)  )/ (z -zp0) )+ 24 ) < ypixel+1 )\
                    )
        # pixelMapConstraint = And(pixelMapConstraint,VertexPixelConstraint1 )
        pixelMapConstraint[i] = VertexPixelConstraint1
       
        # print("checking pos inclusion")
        # scheck =Solver()
        # print(VertexPixelConstraint1)
        # scheck.add(pixelMapConstraint)
        
        # scheck.push()
        # print(scheck.check())
        # print(scheck.model())
        # scheck.add(And(xp0 ==mxp, yp0 == myp, zp0 == mzp))
        # if(scheck.check() != sat):
            
        #     print(" pos not in the region")
        #     print("xcvbnmasdfg")
        #     sleep(5)
    
    
    for i in range(0,numberOfIntersectingEdges):
        # print("___________________")
        # print("current intersecting data ==> ", intersectingEdgeDataToPPL[i] )
        # scheck =Solver()
        # scheck.push()
        # scheck.add(pixelMapConstraint)
        # scheck.add(And(xp0 ==mxp, yp0 == myp, zp0 == mzp))
        # print(scheck.check())
        # print(scheck.model())   
        # scheck.pop()
        
        insideVertex = intersectingEdgeDataToPPL[i][1]
        outsideVertex = intersectingEdgeDataToPPL[i][2]
        
        plane = eval(str(intersectingEdgeDataToPPL[i][3]))
        xpixel = eval(str(intersectingEdgeDataToPPL[i][4]))
        ypixel = eval(str(intersectingEdgeDataToPPL[i][5]))
        
        currCons = symIntersectionPoint(plane,insideVertex, outsideVertex,xpixel,ypixel,xp0,yp0,zp0)
        pixelMapConstraint[i+numberOfFullyInsideVertices] = currCons 
        
        # print(plane,insideVertex, outsideVertex,xpixel,ypixel,xp0,yp0,zp0)
        # print(mxp,myp,mzp)
        # print("checking pos inclusion")
        
        # print(currCons)
        
        # scheck.push()
        # scheck.add(currCons)
        # scheck.add(And(xp0 ==mxp, yp0 == myp, zp0 == mzp))
        # print(scheck.check())
        # print(scheck.model())   
        # scheck.pop()
        # sleep(5)
        # print("\nsecond\n")
        # scheck.add(pixelMapConstraint)
        # scheck.add(And(xp0 ==mxp, yp0 == myp, zp0 == mzp))
        
        # print(scheck.check())
        # print(scheck.model())
        
        # if(scheck.check() != sat):
            
        #     print(" pos not in the region")
        #     print("xcvbnmasdfg")
        #     sleep(5)
    
    finalConstraintToReturn = And(True)
    # print("\n\nFull invariant region computed")
    # print("check pos inclusion")
    scheck =Solver()  
    
    for i in range(0, numberOfFullyInsideVertices+numberOfIntersectingEdges-1):
        # print(i)
        scheck.push()
        scheck.add(pixelMapConstraint[i])
        scheck.add(And(xp0 ==mxp, yp0 == myp, zp0 == mzp))
        # print(scheck.check())
        if(scheck.check() != sat):
            print(pixelMapConstraint[i])
            print(" pos not in the region")
            print("xcvbnmasdfg1111")
            # sleep(5) 
            # exit()
        # print("\n\n") 
        scheck.pop()
        
        finalConstraintToReturn = And(finalConstraintToReturn,pixelMapConstraint[i])
        #finalConstraintToReturn = And(pixelMapConstraint[2],pixelMapConstraint[1])
        
        # print(pixelMapConstraint[i])
        
        s2solver = Solver()
        s2solver.add(finalConstraintToReturn)
        s2solver.add(And(xp0 ==mxp, yp0 == myp, zp0 == mzp))
        # print(s2solver.check())
    
    # finalConstraintToReturn = And(finalConstraintToReturn, Or(And(xp0 ==mxp, yp0 == myp, zp0 == mzp)))
    
    # print("full ANd")
    scheck.add(finalConstraintToReturn)
    scheck.add(And(xp0 ==mxp, yp0 == myp, zp0 == mzp))
    print(scheck.check())
    if(scheck.check() != sat):
            
            print("All pos not in the region")
            print("xcvbnmasdfg2222222222")
            # exit()
            # sleep(5) 
    
    
    # print("waiting...") 
    # sleep(1)
    print("done")
    return finalConstraintToReturn