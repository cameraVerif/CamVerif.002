from z3 import *
import environment


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
        # print("checking intersection with the bottom plane")
           

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
    
    
    
    f1 = p0+q0 == 1
    f2 = And(p0>=0,q0>=0)
    f3 = u0+v0+w0+g0 == 1
    f4 = And(u0>=0, v0>=0, w0>=0,g0>=0)
    
    
    f5 = (p0* (vertices[insideVertex*3+0] - xp0) + q0* (vertices[outsideVertex*3+0] - xp0) )  == (u0*x0+v0*x1+w0*x2+g0*x3)
    f6 = (p0* (vertices[insideVertex*3+1] - yp0) + q0* (vertices[outsideVertex*3+1] - yp0) )  == (u0*y0+v0*y1+w0*y2+g0*y3)
    f7 = (p0* (vertices[insideVertex*3+2] - zp0) + q0* (vertices[outsideVertex*3+2] - zp0) )  == (u0*z0+v0*z1+w0*z2+g0*z3)
    
    
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
    
    pixelMapConstraint = And(True)
    
    
    print("numberOfFullyInsideVertices = ",numberOfFullyInsideVertices)
    for i in range(0,numberOfFullyInsideVertices):        
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
        pixelMapConstraint = And(pixelMapConstraint,VertexPixelConstraint1 )
    
    
    for i in range(0,numberOfIntersectingEdges):
        
        insideVertex = intersectingEdgeDataToPPL[i][1]
        outsideVertex = intersectingEdgeDataToPPL[i][2]
        
        plane = eval(str(intersectingEdgeDataToPPL[i][3]))
        xpixel = eval(str(intersectingEdgeDataToPPL[i][4]))
        ypixel = eval(str(intersectingEdgeDataToPPL[i][5]))
        
        currCons = symIntersectionPoint(plane,insideVertex, outsideVertex,xpixel,ypixel,xp0,yp0,zp0)
        pixelMapConstraint = And(pixelMapConstraint,currCons )
        
    
    return simplify(pixelMapConstraint)