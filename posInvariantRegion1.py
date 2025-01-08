from z3 import *
import random
from time import sleep
import environment
import hashlib
import math
import time
from time import time
from datetime import datetime 


import pyparma_posInvRegion32

vertices = environment.vertices
numOfVertices = environment.numOfVertices
tedges = environment.tedges
numOftedges = environment.numOfEdges

canvasWidth = environment.canvasWidth
canvasHeight = environment.canvasHeight
focalLength = environment.focalLength
t= environment.t
b = environment.b
l = environment.l
r = environment.r
n = environment.n
f = environment.f

outValues = [0]*numOfVertices*4*5

mProj = [
        [2 * n / (r - l), 0, 0, 0],
        [0,2 * n / (t - b),0,0],
        [(r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n), -1 ],
        [0,0,-2 * f * n / (f - n),0]
    ]

xp0,yp0,zp0 = Reals('xp0 yp0 zp0')


    
def getVertexPixelValueIntersectZ3(x,y,z):
    s3 = Solver()
    set_param('parallel.enable', True)
    # set_option(rational_to_decimal=False)
    # set_option(precision=20)
    set_param('parallel.enable', True)
    s3.set("sat.local_search_threads", 28)
    s3.set("sat.threads", 28)
    s3.set("sat.threads", 28)
    s3.check()
    
    a,b = Reals('a b')
    cons1 = ( (((-67*(x ) )/ (z) )+ 24 ) == a )
    cons2 =  ( (((67*(y ) )/ (z) )+ 24 ) == b )
    
    s3.add(simplify(And(cons1,cons2)))
    p = [0,0]
    if s3.check() == sat:
        m = s3.model()
        #print("etVertexPixelValueIntersectZ3: model from solver :",m)
        a = str(eval("m[a].numerator_as_long()/m[a].denominator_as_long()"))
        b = str(eval("m[b].numerator_as_long()/m[b].denominator_as_long()"))
        # print(a,b)   
        a= float(a)
        b= float(b)
        
        frac_a, whole_a = math.modf(a)
        frac_b, whole_b = math.modf(b)
        # print("fractionalPart ", frac, whole)
        
        if(frac_a >= 0.9999):
            p[0] = math.floor(float(a))+1
        else:
            p[0] = math.floor(float(a))
            
        if(frac_a <= 0.0000001):
            p[0] = math.floor(float(a))-1
        else:
            p[0] = math.floor(float(a))
        
        if(frac_b >= 0.9999):
            p[1] = math.floor(float(b))+1
        else:
            p[1] = math.floor(float(b))
        
        if(frac_b <= 0.00000001):
            p[1] = math.floor(float(b))-1
        else:
            p[1] = math.floor(float(b))
        
        # p[0] = math.floor(float(a)) 
        # p[1] = math.floor(float(b))   
        # print("p = ",p)
            
        return p
    else:
        
        exit()  
         

# plane0 = top, plane1 = left, plane2=bottom, plane3= right
def planeEdgeIntersection(plane,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq):
    
    # print("\n\n")
    # print(" Inside planeEdgeIntersection function")
    # print(plane,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
    s1 = Solver()
    set_param('parallel.enable', True)
    # set_option(rational_to_decimal=False)
    # set_option(precision=20)
    set_param('parallel.enable', True)
    s1.set("sat.local_search_threads", 26)
    s1.set("sat.threads", 26)
    s1.set("timeout",10000)
    
   
    #print("inside vertex :", insideVertex)
    #print("outside vertex :", outsideVertex)
    #print(posXp,posYp,posZp)
    
    xv0 = vertices[insideVertex*3+0] - posXp
    yv0 = vertices[insideVertex*3+1] - posYp
    zv0 = vertices[insideVertex*3+2] - posZp
    wv0 = -(vertices[insideVertex*3+2] - posZp)
    
    xv1 = vertices[outsideVertex*3+0] - posXp
    yv1 = vertices[outsideVertex*3+1] - posYp
    zv1 = vertices[outsideVertex*3+2] - posZp
    wv1 =  - (vertices[outsideVertex*3+2] - posZp)
    
    
   
    
    p,q = Reals('p q')
    xl, yl, zl = Reals('xl yl zl')
    
    s1.add(p+q == 1)
    s1.add(And(p>=0,q>=0))
    
    s1.add(xl == (p*xv0+ q*xv1))
    s1.add(yl == (p*yv0+ q*yv1))
    s1.add(zl == (p*zv0+ q*zv1))
    
   
    
    xk, yk, zk = Reals('xk yk zk')
    u, v, w, g  = Reals('u v w g')
    s1.check()
    s1.add(u+v+w+g == 1)
    s1.add(And(u>=0, v>=0, w>=0,g>=0))
    s1.check()
    
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


        
    
    
    
    s1.add(xk == (u*x0+v*x1+w*x2+g*x3))
    s1.add(yk == (u*y0+v*y1+w*y2+g*y3))
    s1.add(zk == (u*z0+v*z1+w*z2+g*z3))
    s1.check()
    # m23 =s1.model()
    # print(m23)

    s1.add(xl == xk)
    s1.add(yl == yk)
    s1.add(zl == zk)
    
    # for c in s1.assertions():
    #     print(c,"\n")
    
    result = s1.check()
    if(result ==sat):
        # print("intersecting with the left plane using linear combination;")
        m = s1.model()
        
        insideFraction = eval("m[p].numerator_as_long()/m[p].denominator_as_long() " )
        outsideFraction = eval("m[q].numerator_as_long()/m[q].denominator_as_long() " )
        
        # vertexPixelValue2 = getVertexPixelValueIntersectZ3(m[p]*xv0+m[q]*xv1,\
        #                                                    m[p]*yv0+m[q]*yv1,\
        #                                                    m[p]*zv0+m[q]*zv1)  
                                
        # print("p, q : ",m[p],m[q])
        # print("vertex 0 :",xv0,yv0,zv0)
        # print("vertex 1 :",xv1,yv1,zv1)
        # print("insideFraction  : ",insideFraction)
        # print("outsideFraction  : ",outsideFraction)
        intersectionPoint[0] =  eval("(1- outsideFraction)*xv0+ outsideFraction*xv1")
        intersectionPoint[1] = eval("(1- outsideFraction)*yv0+ outsideFraction*yv1")
        intersectionPoint[2] = eval("(1- outsideFraction)*zv0+ outsideFraction*zv1")
        intersectionPoint[3] = eval("(1- outsideFraction)*wv0+ outsideFraction*wv1")
        # print("intersection point  using p,q:", intersectionPoint)
        
        # print("\n\n\n point in plane intersection point using u v w")
        mp = m[p]
        mq = m[q]
        # print(m[u]*x0+m[v]*x1+m[w]*x2+m[g]*x3)
        # print(m[u]*y0+m[v]*y1+m[w]*y2+m[g]*y3)
        # print(m[u]*z0+m[v]*z1+m[w]*z2+m[g]*z3)
        # print("\n\n")
        # print("Returning planeEdgeIntersection")
        return 1
    elif result == unsat:
      
        return 0   
    else:
        
        sleep(100)
                         
    
    
def computeOutcodeAtPos(i,outcodeP0, inx, iny,inz):
    
    outcode = 0   
    outx   = inx * mProj[0][0] + iny * mProj[1][0] + inz * mProj[2][0] +  mProj[3][0]
    outy   = inx * mProj[0][1] + iny * mProj[1][1] + inz * mProj[2][1] +  mProj[3][1] 
    outz   = inx * mProj[0][2] + iny * mProj[1][2] + inz * mProj[2][2] +  mProj[3][2] 
    w = inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3] 
    
    global outValues
    outValues[i*4+0] = outx
    outValues[i*4+1] = outy
    outValues[i*4+2] = outz
    outValues[i*4+3] = w
    
    # print(inx, iny, inz, outx, outy, outz,w)

	
	# NFLRBT
    if( not(-w <= outx) ):
        # left
        # outcode=outcode ^ (1 << 3)
        outcodeP0[i*6+3] =1
        # outcode[3] = 1;
    if(not(outx <=w) ):
		# //right
        # outcode=outcode ^ (1 << 2)
        outcodeP0[i*6+2] =1
		# outcode[2] = 1;
    
    if( not(-w <= outy) ):
		# //bottom
        # outcode[1] = 1;
        # outcode=outcode ^ (1 << 1)
        outcodeP0[i*6+1] =1
    if(not(outy <=w) ):
		# //top
        # outcode=outcode ^ (1 << 0)
        outcodeP0[i*6+0] =1
		# outcode[0] = 1;
    
    if( not(-w <= outz) ):
		# //near
        # print("vertex outside near plane")
        # sleep(1)
        # outcode[5] = 1;
        # outcode=outcode ^ (1 << 5)
        outcodeP0[i*6+5] =1
    if(not(outz <=w) ):
		# //far
        # outcode[4] = 1;
        # outcode=outcode ^ (1 << 4)
        outcodeP0[i*6+4] =1


def getVertexPixelValueZ3(xp,yp,zp,x,y,z):
    s3 = Solver()
    set_param('parallel.enable', True)
    set_option(rational_to_decimal=True)
    set_option(precision=20)
    set_param('parallel.enable', True)
    s3.set("sat.local_search_threads", 28)
    s3.set("sat.threads", 28)
    s3.set("sat.threads", 28)
    s3.check()
    
    a,b = Reals('a b')
    cons1 = ( (((-67*(x -xp) )/ (z -zp) )+ 24 ) == a )
    cons2 =  ( (((67*(y -yp) )/ (z -zp) )+ 24 ) == b )
    
    # print(x,y,z)
    
    s3.add(simplify(And(cons1,cons2)))
    p = [0,0]
    if s3.check() == sat:
        m = s3.model()
        #print("model from solver :",m)
        a = str(eval("m[a].numerator_as_long()/m[a].denominator_as_long()"))
        b = str(eval("m[b].numerator_as_long()/m[b].denominator_as_long()"))
        
        # print(a,b)    
        
        # a= float(a)
        # b= float(b)
        
        # frac_a, whole_a = math.modf(a)
        # frac_b, whole_b = math.modf(b)
        # # print("fractionalPart ", frac, whole)
        
        # if(frac_a >= 0.99999999999999999):
        #     p[0] = math.floor(float(a))+1
        # else:
        #     p[0] = math.floor(float(a))
            
        # if(frac_a <= 0.000000000000000001):
        #     p[0] = math.floor(float(a))-1
        # else:
        #     p[0] = math.floor(float(a))
        
        # if(frac_b >= 0.9999999999999999999):
        #     p[1] = math.floor(float(b))+1
        # else:
        #     p[1] = math.floor(float(b))
        
        # if(frac_b <= 0.0000000000000000001):
        #     p[1] = math.floor(float(b))-1
        # else:
        #     p[1] = math.floor(float(b))
        
        p[0] = math.floor(float(a)) 
        p[1] = math.floor(float(b))   
        # print("p = ",p)
        
        
        
        
        # p[0] = math.floor(float(a))
        # p[1] = math.floor(float(b))
        # set_option(rational_to_decimal=False)    
        return p
    else:
        print("no sat image")
        exit()  

def computePosInvariantRegion(posXp, posYp, posZp, m, currRegionPPL ):
 
    currImage =[]
    
    outcodeP0 = [0]*numOfVertices*6
    for i in range(0, numOfVertices):
        # print(i)
        computeOutcodeAtPos(i,outcodeP0, vertices[i*3+0] -posXp ,vertices[i*3+1]-posYp,vertices[i*3+2]-posZp )
        
        
        # 0:fully inside,  1: intersect, 2: fully outside
        edgesOutInInter =[0]*numOftedges
        
        print("classify edges inside, outside, intersec :start")
        for j in range(0,numOftedges):
            # print("edge : ",j)
            edge_v0 = tedges[j*2+0]
            edge_v1 = tedges[j*2+1]
            # print(edge_v0,edge_v1)
            
            # check whether the edge intersect, fullyoutside, or inside
            if( outcodeP0[edge_v0*6+0] == 0 and outcodeP0[edge_v0*6+1] == 0  and outcodeP0[edge_v0*6+2] == 0 and \
                    outcodeP0[edge_v0*6+3] == 0 and outcodeP0[edge_v0*6+4] == 0 and outcodeP0[edge_v0*6+5] == 0 and \
                    outcodeP0[edge_v1*6+0] == 0 and outcodeP0[edge_v1*6+1] == 0  and outcodeP0[edge_v1*6+2] == 0 and \
                    outcodeP0[edge_v1*6+3] == 0 and outcodeP0[edge_v1*6+4] == 0 and outcodeP0[edge_v1*6+5] == 0 ):

                edgesOutInInter[j] = 0
            elif((outcodeP0[edge_v0*6+0] == 1 and outcodeP0[edge_v1*6+0] ==1) or\
                    (outcodeP0[edge_v0*6+1] == 1 and outcodeP0[edge_v1*6+1] ==1) or \
                    (outcodeP0[edge_v0*6+2] == 1 and outcodeP0[edge_v1*6+2] ==1) or \
                    (outcodeP0[edge_v0*6+3] == 1 and outcodeP0[edge_v1*6+3] ==1) or \
                    (outcodeP0[edge_v0*6+4] == 1 and outcodeP0[edge_v1*6+4] ==1 ) or\
                    (outcodeP0[edge_v0*6+5] == 1 and outcodeP0[edge_v1*6+5] ==1)):
                edgesOutInInter[j] = 2
            else:
                edgesOutInInter[j] = 1
            
            # print(j," : ",edgesOutInInter[j])
            
        newvertices = [0]*numOfVertices*3*5
        newVerticesNumber = 0
        pixelValueComputed = [0]*numOfVertices*5
        pixelValues = [0]*numOfVertices*2*5
        
        edgesInSmallPyramid = [0]*numOftedges
        
        insideVertexDetailsToPPL = [] #store vertex index number,xpixel,ypixel
        numberOfFullyInsideVertices = 0
        numberOfIntersectingEdges = 0
        intersectingEdgeDataToPPL = []
        
        pixelMapCons = And(True)
        
       
        for j in range(0,numOftedges):
            if(j %25 == 0):
                print(j)

            if(edgesOutInInter[j] == 0):
                # fully inside
                # print("fully inside edge")
                edge_v0 = tedges[j*2+0]
                edge_v1 = tedges[j*2+1] 
                
                if(pixelValueComputed[edge_v0] == 0):
                    pixelValueComputed[edge_v0] = 1
                    x = vertices[edge_v0*3+0]
                    y = vertices[edge_v0*3+1]
                    z = vertices[edge_v0*3+2]
                    vertexPixelValue = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
                    # print("edge_v0: ", edge_v0," pixelvalue : ", vertexPixelValue)
                    # vertexPixelValue =getVertexPixelValuePython(posXp,posYp,posZp,x,y,z)
                    # vertexPixelValue = getVertexPixelValueZ3(posXp,posYp,posZp,x,y,z)
                    # add the vertex details to PPL ds
                    if(vertexPixelValue[0] >=0 and vertexPixelValue[1]>=0):
                        tempPixelData = [edge_v0, vertexPixelValue[0],vertexPixelValue[1]]
                        insideVertexDetailsToPPL.append(tempPixelData)
                        numberOfFullyInsideVertices+=1
                        currImage.append([vertexPixelValue[0],vertexPixelValue[1]])
                        # print("edge_v0: ", edge_v0," pixelvalue : ", vertexPixelValue)
                    
            
                    
                
                if(pixelValueComputed[edge_v1] == 0):
                    pixelValueComputed[edge_v1] = 1
                    x = vertices[edge_v1*3+0]
                    y = vertices[edge_v1*3+1]
                    z = vertices[edge_v1*3+2]
                    vertexPixelValue = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
                    # print("insideVertex a: ", insideVertex," pixelvalue : ", vertexPixelValue)
                    # vertexPixelValue =getVertexPixelValuePython(posXp,posYp,posZp,x,y,z)
                    # print("insideVertex b: ", insideVertex," pixelvalue : ", vertexPixelValue)
                    # vertexPixelValue = getVertexPixelValueZ3(posXp,posYp,posZp,x,y,z)
                    # add the vertex details to PPL ds
                    
                    if(vertexPixelValue[0] >=0 and vertexPixelValue[1]>=0):
                        tempPixelData = [edge_v1, vertexPixelValue[0],vertexPixelValue[1]]
                        insideVertexDetailsToPPL.append(tempPixelData)
                        numberOfFullyInsideVertices+=1
                        currImage.append([vertexPixelValue[0],vertexPixelValue[1]])
                        # print("edge_v1: ", edge_v1," pixelvalue : ", vertexPixelValue)
                    
                    
              
                    
            if(edgesOutInInter[j] == 1):
                # edges intersecting
                # print("intersecting edge")
                # edge_v0 = tedges[j*2+0]
                # edge_v1 = tedges[j*2+1] 
                # edge_v0 = tedges[(4*9*2)+ (currTriangle-16) *6+j*2+0]
                # edge_v1 = tedges[(4*9*2)+ (currTriangle-16) *6+j*2+1] 
                # edge_v0 = 0
                # edge_v1 = 0
                # if (currTriangle >=16):
                #     edge_v0 = tedges[(4*9*2)+ (currTriangle-16) *6+j*2+0]
                #     edge_v1 = tedges[(4*9*2)+ (currTriangle-16) *6+j*2+1] 
                # else:
                #     if (j == 0):                
                #         edge_v0 = vertex0
                #         edge_v1 = vertex1
                #     elif (j==1):    
                #         edge_v0 = vertex1
                #         edge_v1 = vertex2
                #     else:
                #         edge_v0 = vertex2
                #         edge_v1 = vertex0
                
                # print(edge_v0,edge_v1)
                
                edge_v0 = tedges[j*2+0]
                edge_v1 = tedges[j*2+1] 

                outsideVertex = edge_v1
                insideVertex  = edge_v0
                
                sumOfOutcode_v0 = outcodeP0[edge_v0*6+0]+outcodeP0[edge_v0*6+1] +outcodeP0[edge_v0*6+2]+\
                                 outcodeP0[edge_v0*6+3]+outcodeP0[edge_v0*6+4] +outcodeP0[edge_v0*6+5]
                sumOfOutcode_v1 = outcodeP0[edge_v1*6+0]+outcodeP0[edge_v1*6+1] +outcodeP0[edge_v1*6+2]+\
                    outcodeP0[edge_v1*6+3]+outcodeP0[edge_v1*6+4] +outcodeP0[edge_v1*6+5]
                
                bothVerticesOutside = 0
                if ( sumOfOutcode_v0== 0 and sumOfOutcode_v1 >0):
                    insideVertex  = edge_v0
                    outsideVertex = edge_v1
                elif(sumOfOutcode_v0 > 0 and sumOfOutcode_v1 == 0):
                    insideVertex  = edge_v1
                    outsideVertex = edge_v0
                else:
                    bothVerticesOutside = 1
                
                if(bothVerticesOutside == 0 ):
                    # print("one vertex is inside")
                    # print("\n\nintersecting edge = ", j)
                    # print("insideVertex = ",insideVertex)
                    if(pixelValueComputed[insideVertex] == 0):
                        pixelValueComputed[insideVertex] = 1
                        x = vertices[insideVertex*3+0]
                        y = vertices[insideVertex*3+1]
                        z = vertices[insideVertex*3+2]
                        vertexPixelValue = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
                        # print("insideVertex a: ", insideVertex," pixelvalue : ", vertexPixelValue)
                        # vertexPixelValue =getVertexPixelValuePython(posXp,posYp,posZp,x,y,z)
                        # print("insideVertex b: ", insideVertex," pixelvalue : ", vertexPixelValue)
                        #vertexPixelValue = getVertexPixelValueZ3(posXp,posYp,posZp,x,y,z)
                        if(vertexPixelValue[0] >=0 and vertexPixelValue[1]>=0):
                            tempPixelData = [insideVertex, vertexPixelValue[0],vertexPixelValue[1]]
                            insideVertexDetailsToPPL.append(tempPixelData)
                            numberOfFullyInsideVertices+=1
                            currImage.append([vertexPixelValue[0],vertexPixelValue[1]])
                            # print("insideVertex: ", insideVertex," pixelvalue : ", vertexPixelValue)
                        
                    
                    
                
                    
                   
                    if(0 == 1):
                        # print("outside vertex is inside the small pyramid")
                        if(pixelValueComputed[outsideVertex] == 0):
                            pixelValueComputed[outsideVertex] = 1
                            x = vertices[outsideVertex*3+0]
                            y = vertices[outsideVertex*3+1]
                            z = vertices[outsideVertex*3+2]
                            vertexPixelValue = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
                            #vertexPixelValue =getVertexPixelValuePython(posXp,posYp,posZp,x,y,z)
                            #vertexPixelValue = getVertexPixelValueZ3(posXp,posYp,posZp,x,y,z)
                            if(vertexPixelValue[0] >=0 and vertexPixelValue[1]>=0):
                                tempPixelData = [outsideVertex, vertexPixelValue[0],vertexPixelValue[1]]
                                insideVertexDetailsToPPL.append(tempPixelData)
                                numberOfFullyInsideVertices+=1
                                currImage.append([vertexPixelValue[0],vertexPixelValue[1]])
                                # print("outside vertex : ",outsideVertex, " pixel value :",vertexPixelValue)
                            
                          
                        
                        
                    else:
                        # print("outside vertex not in small pyramid")
                        # print("check for intersection with four planes")
                        
                     
                        # find value of intersection point for the current xp,yp,zp
                        intersectionPoint = [0,0,0,0]
                        vertexPixelValue2 = [0,0]
                        mp = 0
                        mq = 0
                        
                        if(outcodeP0[outsideVertex*6+0] != 0):
                            # top plane
                            # print("top plane")
                            # planeEdgeIntersectionPython(0,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                            # print(vertexPixelValue2)  
                            #getPixelValueIntersectionPointZ3(m[xp0],m[yp0],m[zp0],x,y,z)
                            #if( (vertexPixelValue2[0]>=0 and vertexPixelValue2[0]<=48) and (vertexPixelValue2[1]==0)) : 
                            if(planeEdgeIntersection(0,insideVertex, outsideVertex,m[xp0],m[yp0],m[zp0],intersectionPoint,vertexPixelValue2,mp,mq) == 1):
                                if( (vertexPixelValue2[0]>=0 and vertexPixelValue2[0]<=48) and (vertexPixelValue2[1]==0)) : 
                                    # if( (vertexPixelValue2[0]==0 and vertexPixelValue2[1]==0)):
                                    #     print("both zero pixels using the solver")
                                    #     planeEdgeIntersection(0,insideVertex, outsideVertex,m[xp0],m[yp0],m[zp0],intersectionPoint,vertexPixelValue,mp,mq)
                                    #     print(vertexPixelValue)
                                    #     if(vertexPixelValue[0]<0):
                                    #         continue
                                        
                                        # print(intersectionPoint)
                                        # print(vertexPixelValue2)
                                        # continue
                                    
                                    
                                    # print("intersecting with top plane")
                                    currentIntersectingPlane =0
                                    # print(intersectionPoint)
                                    # print(intersectionPoint)
                                    # planeEdgeIntersectionPython(0,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                                    # print(intersectionPoint)
                                    # x = intersectionPoint[0]
                                    # y = intersectionPoint[1]
                                    # z = intersectionPoint[2]
                                    
                                    # vertexPixelValue = getPixelValueIntersectionPoint(posXp,posYp,posZp,x,y,z)
                                    # vertexPixelValue = getPixelValueIntersectionPointPython(posXp,posYp,posZp,x,y,z)
                                    # print("Intersecting point :", intersectionPoint," pixel Value :",vertexPixelValue)
                                    # print(vertexPixelValue2)
                                    # print(vertexPixelValue)
                                    # print("\n")
                                    
                                    # planeEdgeIntersectionPython(0,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                                    x = intersectionPoint[0]
                                    y = intersectionPoint[1]
                                    z = intersectionPoint[2]
                                    # intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                    #                     math.floor(vertexPixelValue[0]), math.floor(vertexPixelValue[1]),x,y,z,mp,mq]
                                    # if(vertexPixelValue2[0] > 0):
                                    #         xpixel =vertexPixelValue2[0]-1
                                    # else:
                                    #     xpixel =0
                                
                                    # if(vertexPixelValue2[1] < 48):
                                    #     ypixel =vertexPixelValue2[1]-1
                                    # else:
                                    #     ypixel =48  
                                    vertexPixelValue2 = getVertexPixelValueIntersectZ3(x,y,z)
                                    xpixel = vertexPixelValue2[0]
                                    ypixel = vertexPixelValue2[1]
                                    print(xpixel,ypixel)
                                    
                                    if(xpixel<0 or ypixel <0):
                                        continue
                                    
                                    intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                                        xpixel,ypixel,x,y,z,mp,mq]
                                    #print("intersecting data :",intersectingData)
                                    intersectingEdgeDataToPPL.append(intersectingData)
                                    numberOfIntersectingEdges += 1
                                    currImage.append([math.floor(float(vertexPixelValue2[0])),\
                                        math.floor(float(vertexPixelValue2[1]))])
                        
                        if(outcodeP0[outsideVertex*6+1] != 0):
                            # print("\nbottom plane_")
                            #planeEdgeIntersectionPython(1,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                            # print(vertexPixelValue2)
                            #if( (vertexPixelValue2[0]>=0 and vertexPixelValue2[0]<=48) and ( vertexPixelValue2[1]==48)) :
                            if(planeEdgeIntersection(1,insideVertex, outsideVertex,m[xp0],m[yp0],m[zp0],intersectionPoint,vertexPixelValue2,mp,mq) == 1):
                                if( (vertexPixelValue2[0]>=0 and vertexPixelValue2[0]<=48) and ( vertexPixelValue2[1]==48)) :
                                    # print("intersecting with bottom plane")
                                    
                                    currentIntersectingPlane =1
                                    # print(intersectionPoint)
                                    # print(intersectionPoint)
                                    # planeEdgeIntersectionPython(1,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                                    # print(intersectionPoint)
                                    # x = intersectionPoint[0]
                                    # y = intersectionPoint[1]
                                    # z = intersectionPoint[2]
                                    
                                    # # vertexPixelValue = getPixelValueIntersectionPointPython(posXp,posYp,posZp,x,y,z)
                                    # vertexPixelValue = getPixelValueIntersectionPointPython(posXp,posYp,posZp,x,y,z)
                                    # print("Intersecting point :", intersectionPoint," pixel Value :",vertexPixelValue)
                                    # print(vertexPixelValue2)
                                    # print(vertexPixelValue)
                                    
                                    # print(vertexPixelValue2)
                                    # # print(vertexPixelValue)
                                    # print("\n")
                                    
                                    
                                    
                                    # planeEdgeIntersectionPython(1,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                                    # vertexPixelValue2 = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
                                    x = intersectionPoint[0]
                                    y = intersectionPoint[1]
                                    z = intersectionPoint[2]
                                    
                                    
                                    # if(vertexPixelValue2[0] > 0):
                                    #         xpixel =vertexPixelValue2[0]-1
                                    # else:
                                    #     xpixel =0
                                
                                    # if(vertexPixelValue2[1] < 48):
                                    #     ypixel =vertexPixelValue2[1]-1
                                    # else:
                                    #     ypixel =48  
                                    vertexPixelValue2 = getVertexPixelValueIntersectZ3(x,y,z)
                                    
                                    xpixel = vertexPixelValue2[0]
                                    ypixel = vertexPixelValue2[1]
                                    # print("adding ppl datat1")
                                    # print("xpixel, ypixel")
                                    # print(xpixel,ypixel)
                                    if(xpixel<0 or ypixel <0):
                                        continue
                                    # intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                    #                     math.floor(vertexPixelValue[0]), math.floor(vertexPixelValue[1]),x,y,z,mp,mq]
                                    intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                                        xpixel,ypixel,x,y,z,mp,mq]
                                    #print("intersecting data :",intersectingData)
                                    intersectingEdgeDataToPPL.append(intersectingData)
                                    numberOfIntersectingEdges += 1
                                    currImage.append([math.floor(float(vertexPixelValue2[0])),\
                                        math.floor(float(vertexPixelValue2[1]))])
                        
                        if(outcodeP0[outsideVertex*6+2] != 0):  
                            # print("right plane")
                            #planeEdgeIntersectionPython(2,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                            # print(vertexPixelValue2)
                            #if( ( vertexPixelValue2[0]==48) and (vertexPixelValue2[1]>=0 and vertexPixelValue2[1]<=48)) : 
                            if(planeEdgeIntersection(2,insideVertex, outsideVertex,m[xp0],m[yp0],m[zp0],intersectionPoint,vertexPixelValue2,mp,mq) == 1):
                                if( ( vertexPixelValue2[0]==48) and (vertexPixelValue2[1]>=0 and vertexPixelValue2[1]<=48)) : 
                                    # print("intersecting with right plane")
                                    currentIntersectingPlane =2
                                    # print(intersectionPoint)
                                    # print(intersectionPoint)
                                    # planeEdgeIntersectionPython(2,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                                    # print(intersectionPoint)
                                    # x = intersectionPoint[0]
                                    # y = intersectionPoint[1]
                                    # z = intersectionPoint[2]
                                    
                                    # # vertexPixelValue = getPixelValueIntersectionPoint(posXp,posYp,posZp,x,y,z)
                                    # vertexPixelValue = getPixelValueIntersectionPointPython(posXp,posYp,posZp,x,y,z)
                                    # print("Intersecting point :", intersectionPoint," pixel Value :",vertexPixelValue)
                                    # print(vertexPixelValue2)
                                    # print(vertexPixelValue)
                                    
                                    # print(vertexPixelValue2)
                                    # print(vertexPixelValue)
                                    # print("\n")
                                    
                                    
                                    # planeEdgeIntersectionPython(2,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                                    
                                    x = intersectionPoint[0]
                                    y = intersectionPoint[1]
                                    z = intersectionPoint[2]
                                    
                                    # if(vertexPixelValue2[0] > 0):
                                    #         xpixel =vertexPixelValue2[0]-1
                                    # else:
                                    #     xpixel =0
                                
                                    # if(vertexPixelValue2[1] < 48):
                                    #     ypixel =vertexPixelValue2[1]-1
                                    # else:
                                    #     ypixel =48  
                                    # vertexPixelValue2 = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
                                    
                                    vertexPixelValue2 = getVertexPixelValueIntersectZ3(x,y,z)
                                    xpixel = vertexPixelValue2[0]
                                    ypixel = vertexPixelValue2[1]
                                    # print(xpixel,ypixel)
                                    if(xpixel<0 or ypixel <0):
                                        continue
                                    
                                    # intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                    #                     math.floor(vertexPixelValue[0]), math.floor(vertexPixelValue[1]),x,y,z,mp,mq]
                                    intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                                        xpixel,ypixel,x,y,z,mp,mq]
                                    
                                    #print("intersecting data :",intersectingData)
                                    intersectingEdgeDataToPPL.append(intersectingData)
                                    numberOfIntersectingEdges += 1
                                    currImage.append([math.floor(float(vertexPixelValue2[0])),\
                                        math.floor(float(vertexPixelValue2[1]))])
                            
                        if(outcodeP0[outsideVertex*6+3] != 0):
                            # print("left plane__")
                            #planeEdgeIntersectionPython(3,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                            # print(vertexPixelValue2)
                            
                            #if( (vertexPixelValue2[0]==0 ) and (vertexPixelValue2[1]>=0 and vertexPixelValue2[1]<=48)) :
                            if(planeEdgeIntersection(3,insideVertex, outsideVertex,m[xp0],m[yp0],m[zp0],intersectionPoint,vertexPixelValue2,mp,mq) == 1):
                                if( (vertexPixelValue2[0]==0 ) and (vertexPixelValue2[1]>=0 and vertexPixelValue2[1]<=48)) :
                                    # print("intersecting with left plane")
                                    currentIntersectingPlane =3
                                    
                                    
                                    # print(intersectionPoint)
                                    # planeEdgeIntersectionPython(3,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                                    # print(intersectionPoint)
                                    # x = intersectionPoint[0]
                                    # y = intersectionPoint[1]
                                    # z = intersectionPoint[2]
                                    
                                    # # vertexPixelValue = getPixelValueIntersectionPoint(posXp,posYp,posZp,x,y,z)
                                    # vertexPixelValue = getPixelValueIntersectionPointPython(posXp,posYp,posZp,x,y,z)
                                    # print("Intersecting point :", intersectionPoint," pixel Value :",vertexPixelValue)
                                    # print(vertexPixelValue2)
                                    # print(vertexPixelValue)
                                    
                                    # print(vertexPixelValue2)
                                    # print(vertexPixelValue)
                                    # print("\n")
                                    
                                    # if(vertexPixelValue2[0] > 0):
                                    #     xpixel =vertexPixelValue2[0]-1
                                    # else:
                                    #     xpixel =0
                                
                                    # if(vertexPixelValue2[1] < 48):
                                    #     ypixel =vertexPixelValue2[1]-1
                                    # else:
                                    #     ypixel =48    
                                        
                                    # planeEdgeIntersectionPython(3,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                                    x = intersectionPoint[0]
                                    y = intersectionPoint[1]
                                    z = intersectionPoint[2]
                                    # vertexPixelValue2 = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)  
                                    vertexPixelValue2 = getVertexPixelValueIntersectZ3(x,y,z)  
                                    xpixel = vertexPixelValue2[0]
                                    ypixel = vertexPixelValue2[1]
                                    # print("adding ppl datat1")
                                    # print("xpixel, ypixel")
                                    # print(xpixel,ypixel)
                                    if(xpixel<0 or ypixel <0):
                                        continue
                                    # intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                    #                     math.floor(vertexPixelValue[0]), math.floor(vertexPixelValue[1]),x,y,z,mp,mq]
                                    intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                                        xpixel, ypixel,x,y,z,mp,mq]
                                    
                                    #print("intersecting data :",intersectingData)
                                    intersectingEdgeDataToPPL.append(intersectingData)
                                    numberOfIntersectingEdges += 1
                                    currImage.append([math.floor(float(vertexPixelValue2[0])),\
                                        math.floor(float(vertexPixelValue2[1]))])
                        
                        if(outcodeP0[outsideVertex*6+5] != 0):    
                            
                            #planeEdgeIntersectionPython(5,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                            # print(vertexPixelValue2)
                            
                            #if( (vertexPixelValue2[0]>=0 and vertexPixelValue2[0]<=48) and (vertexPixelValue2[1]>=0 and vertexPixelValue2[1]<=48)) :
                            
                            if(planeEdgeIntersection(5,insideVertex, outsideVertex,m[xp0],m[yp0],m[zp0],intersectionPoint,vertexPixelValue2,mp,mq) == 1):
                                if( (vertexPixelValue2[0]>=0 and vertexPixelValue2[0]<=48) and (vertexPixelValue2[1]>=0 and vertexPixelValue2[1]<=48)) :
                                    print("intersecting with near plane")
                                    currentIntersectingPlane =5
                                    # print(intersectionPoint)
                                    x = intersectionPoint[0]
                                    y = intersectionPoint[1]
                                    z = intersectionPoint[2]
                                    
                                    # vertexPixelValue = getPixelValueIntersectionPoint(posXp,posYp,posZp,x,y,z)
                                    # vertexPixelValue = getPixelValueIntersectionPointPython(posXp,posYp,posZp,x,y,z)
                                    # print("Intersecting point :", intersectionPoint," pixel Value :",vertexPixelValue)
                                    # print(vertexPixelValue2)
                                    # print(vertexPixelValue)
                                    
                                    # planeEdgeIntersectionPython(5,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                                    # print(vertexPixelValue2)
                                    # print(vertexPixelValue)
                                    # print("\n")
                                    # vertexPixelValue2 = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
                                    vertexPixelValue2 = getVertexPixelValueIntersectZ3(x,y,z)
                                    # intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                    #                     math.floor(vertexPixelValue[0]), math.floor(vertexPixelValue[1]),x,y,z,mp,mq]
                                    intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                                        vertexPixelValue2[0], vertexPixelValue2[1],x,y,z,mp,mq]
                                    
                                    #print("intersecting data :",intersectingData)
                                    intersectingEdgeDataToPPL.append(intersectingData)
                                    numberOfIntersectingEdges += 1
                                    currImage.append([math.floor(float(vertexPixelValue[0])),\
                                        math.floor(float(vertexPixelValue[1]))])
                      
                    
                elif(bothVerticesOutside == 1 ):
                    # print("both vertices are outside the viewing volume, and the edge not in small pyramid")
                    
                    # find value of intersection point for the current xp,yp,zp
                    intersectionPoint = [0,0,0,0]
                    vertexPixelValue2 = [0,0]
                    mp =0
                    mq =0
                    
                    
                    if(outcodeP0[outsideVertex*6+0] != 0  or (outcodeP0[insideVertex*6+0] != 0) ): 
                        # top plane
                        #planeEdgeIntersectionPython(0,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                        # print(vertexPixelValue2)
                        
                        #if( (vertexPixelValue2[0]>=0 and vertexPixelValue2[0]<=48) and (vertexPixelValue2[1]==0 )) :
                          
                    
                        # if(planeEdgeIntersection(0,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint) == 1):
                        if(planeEdgeIntersection(0,insideVertex, outsideVertex,m[xp0],m[yp0],m[zp0],intersectionPoint,vertexPixelValue2,mp,mq) == 1):
                            if( (vertexPixelValue2[0]>=0 and vertexPixelValue2[0]<=48) and (vertexPixelValue2[1]==0 )) :
                                # print("intersecting with top plane")
                                currentIntersectingPlane =0
                                # print(intersectionPoint)
                                # x = intersectionPoint[0]
                                # y = intersectionPoint[1]
                                # z = intersectionPoint[2]
                                
                                # vertexPixelValue = getPixelValueIntersectionPoint(posXp,posYp,posZp,x,y,z)
                                # vertexPixelValue = getPixelValueIntersectionPointPython(posXp,posYp,posZp,x,y,z)
                                # print("Intersecting point :", intersectionPoint," pixel Value :",vertexPixelValue)
                                # print(vertexPixelValue2)
                                # print(vertexPixelValue)
                                
                                
                                
                                x = intersectionPoint[0]
                                y = intersectionPoint[1]
                                z = intersectionPoint[2]
                                
                                # vertexPixelValue2 = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
                                vertexPixelValue2 = getVertexPixelValueIntersectZ3(x,y,z)
                                xpixel = vertexPixelValue2[0]
                                ypixel = vertexPixelValue2[1]
                                # print(xpixel,ypixel)
                                if(xpixel<0 or ypixel <0):
                                        continue
                                    
                                intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                                        xpixel, ypixel,x,y,z,mp,mq]
                                                            
                                # intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                #                     math.floor(vertexPixelValue[0]), math.floor(vertexPixelValue[1]),x,y,z,mp,mq]
                                # intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                #                         vertexPixelValue[0], vertexPixelValue[1],x,y,z,mp,mq]
                                
                                #print("intersecting data :",intersectingData)
                                intersectingEdgeDataToPPL.append(intersectingData)
                                numberOfIntersectingEdges += 1
                                currImage.append([math.floor(float(vertexPixelValue2[0])),\
                                        math.floor(float(vertexPixelValue2[1]))])
                            
                    
                    if(outcodeP0[outsideVertex*6+1] != 0  or (outcodeP0[insideVertex*6+1] != 0) ): 
                        
                        #planeEdgeIntersectionPython(1,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                        # print(vertexPixelValue2)
                        
                        #if( (vertexPixelValue2[0]>=0 and vertexPixelValue2[0]<=48) and ( vertexPixelValue2[1]==48)) :
                          
                    
                        
                        
                        if(planeEdgeIntersection(1,insideVertex, outsideVertex,m[xp0],m[yp0],m[zp0],intersectionPoint,vertexPixelValue2,mp,mq) == 1):
                            if( (vertexPixelValue2[0]>=0 and vertexPixelValue2[0]<=48) and ( vertexPixelValue2[1]==48)) :
                                
                                # print("intersecting with bottom plane")
                                currentIntersectingPlane =1
                                # print(intersectionPoint)
                                # x = intersectionPoint[0]
                                # y = intersectionPoint[1]
                                # z = intersectionPoint[2]
                                
                                # # vertexPixelValue = getPixelValueIntersectionPoint(posXp,posYp,posZp,x,y,z)
                                # vertexPixelValue = getPixelValueIntersectionPointPython(posXp,posYp,posZp,x,y,z)
                                # print("Intersecting point :", intersectionPoint," pixel Value :",vertexPixelValue)
                                # print(vertexPixelValue2)
                                # print(vertexPixelValue)
                                
                                # intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                #                     math.floor(vertexPixelValue[0]), math.floor(vertexPixelValue[1]),x,y,z,mp,mq]
                                
                                
                                # planeEdgeIntersectionPython(1,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                                x = intersectionPoint[0]
                                y = intersectionPoint[1]
                                z = intersectionPoint[2]
                                # vertexPixelValue2 = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
                                vertexPixelValue2 = getVertexPixelValueIntersectZ3(x,y,z)
                                xpixel = vertexPixelValue2[0]
                                ypixel = vertexPixelValue2[1]
                                # print(xpixel,ypixel)
                                if(xpixel<0 or ypixel <0):
                                        continue
                                intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                                            xpixel, ypixel,x,y,z,mp,mq]
                             
                                
                                # intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                #                         vertexPixelValue[0], vertexPixelValue[1],x,y,z,mp,mq]
                                
                                #print("intersecting data :",intersectingData)
                                intersectingEdgeDataToPPL.append(intersectingData)
                                numberOfIntersectingEdges += 1
                                currImage.append([math.floor(float(vertexPixelValue2[0])),\
                                        math.floor(float(vertexPixelValue2[1]))])
                    
                    if(outcodeP0[outsideVertex*6+2] != 0  or (outcodeP0[insideVertex*6+2] != 0) ): 
                        
                        #planeEdgeIntersectionPython(2,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                        # print(vertexPixelValue2)
                        
                        #if( (vertexPixelValue2[0]==48) and (vertexPixelValue2[1]>=0 and vertexPixelValue2[1]<=48)) :
                          
                        
                        if(planeEdgeIntersection(2,insideVertex, outsideVertex,m[xp0],m[yp0],m[zp0],intersectionPoint,vertexPixelValue2,mp,mq) == 1):
                            if( (vertexPixelValue2[0]==48) and (vertexPixelValue2[1]>=0 and vertexPixelValue2[1]<=48)) :
                                # print("intersecting with right plane")
                                currentIntersectingPlane =2
                                # print(intersectionPoint)
                                # x = intersectionPoint[0]
                                # y = intersectionPoint[1]
                                # z = intersectionPoint[2]
                                
                                # # vertexPixelValue = getPixelValueIntersectionPoint(posXp,posYp,posZp,x,y,z)
                                # vertexPixelValue = getPixelValueIntersectionPointPython(posXp,posYp,posZp,x,y,z)
                                # print("Intersecting point :", intersectionPoint," pixel Value :",vertexPixelValue)
                                # print(vertexPixelValue2)
                                # print(vertexPixelValue)
                                
                                # intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                #                     math.floor(vertexPixelValue[0]), math.floor(vertexPixelValue[1]),x,y,z,mp,mq]
                                
                                # planeEdgeIntersectionPython(2,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                                x = intersectionPoint[0]
                                y = intersectionPoint[1]
                                z = intersectionPoint[2]
                                # vertexPixelValue2 = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
                                vertexPixelValue2 = getVertexPixelValueIntersectZ3(x,y,z)
                                xpixel = vertexPixelValue2[0]
                                ypixel = vertexPixelValue2[1]
                                # print(xpixel,ypixel)
                                if(xpixel<0 or ypixel <0):
                                        continue
                                intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                                            xpixel, ypixel,x,y,z,mp,mq]
                                
                                
                                
                                
                                
                                # intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                #                         vertexPixelValue[0], vertexPixelValue[1],x,y,z,mp,mq]
                                
                                #print("intersecting data :",intersectingData)
                                intersectingEdgeDataToPPL.append(intersectingData)
                                numberOfIntersectingEdges += 1
                                currImage.append([math.floor(float(vertexPixelValue2[0])),\
                                        math.floor(float(vertexPixelValue2[1]))])
                        
                    
                    if(outcodeP0[outsideVertex*6+3] != 0  or (outcodeP0[insideVertex*6+3] != 0) ):
                        
                        #planeEdgeIntersectionPython(3,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                        # print(vertexPixelValue2)
                        
                        #if( (vertexPixelValue2[0]==0 ) and (vertexPixelValue2[1]>=0 and vertexPixelValue2[1]<=48)) :
                          
                    
                        
                             
                        if(planeEdgeIntersection(3,insideVertex, outsideVertex,m[xp0],m[yp0],m[zp0],intersectionPoint,vertexPixelValue2,mp,mq) == 1):
                            if( (vertexPixelValue2[0]==0 ) and (vertexPixelValue2[1]>=0 and vertexPixelValue2[1]<=48)) :
                                # print("intersecting with left plane")
                                currentIntersectingPlane =3
                                # print(intersectionPoint)
                                # x = intersectionPoint[0]
                                # y = intersectionPoint[1]
                                # z = intersectionPoint[2]
                                
                                # # vertexPixelValue = getPixelValueIntersectionPoint(posXp,posYp,posZp,x,y,z)
                                # vertexPixelValue = getPixelValueIntersectionPointPython(posXp,posYp,posZp,x,y,z)
                                # print("Intersecting point :", intersectionPoint," pixel Value :",vertexPixelValue)
                                # print(vertexPixelValue2)
                                # print(vertexPixelValue)
                                
                                # intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                #                     math.floor(vertexPixelValue[0]), math.floor(vertexPixelValue[1]),x,y,z,mp,mq]
                                
                                # planeEdgeIntersectionPython(3,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                                x = intersectionPoint[0]
                                y = intersectionPoint[1]
                                z = intersectionPoint[2]
                                # vertexPixelValue2 = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
                                vertexPixelValue2 = getVertexPixelValueIntersectZ3(x,y,z)
                                xpixel = vertexPixelValue2[0]
                                ypixel = vertexPixelValue2[1]
                                # print(xpixel,ypixel)
                                if(xpixel<0 or ypixel <0):
                                        continue
                                intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                                            xpixel, ypixel,x,y,z,mp,mq]
                                
                                
                                
                                
                                
                                
                                # intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                #                         vertexPixelValue[0], vertexPixelValue[1],x,y,z,mp,mq]
                                
                                
                                #print("intersecting data :",intersectingData)
                                intersectingEdgeDataToPPL.append(intersectingData)
                                numberOfIntersectingEdges += 1
                                currImage.append([math.floor(float(vertexPixelValue2[0])),\
                                        math.floor(float(vertexPixelValue2[1]))])
                        
                    if(outcodeP0[outsideVertex*6+5] != 0  or (outcodeP0[insideVertex*6+5] != 0) ):    
                        
                        #planeEdgeIntersectionPython(5,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                        # print(vertexPixelValue2)
                        
                        #if( (vertexPixelValue2[0]>=0 and vertexPixelValue2[0]<=48) and (vertexPixelValue2[1]>=0 and vertexPixelValue2[1]<=48)) :
                          
                    
                        
                             
                        if(planeEdgeIntersection(5,insideVertex, outsideVertex,m[xp0],m[yp0],m[zp0],intersectionPoint,vertexPixelValue2,mp,mq) == 1):
                            if( (vertexPixelValue2[0]>=0 and vertexPixelValue2[0]<=48) and (vertexPixelValue2[1]>=0 and vertexPixelValue2[1]<=48)) :
                                # print("intersecting with near plane")
                                currentIntersectingPlane =5
                                # print(intersectionPoint)
                                # x = intersectionPoint[0]
                                # y = intersectionPoint[1]
                                # z = intersectionPoint[2]
                                
                                # # vertexPixelValue = getPixelValueIntersectionPoint(posXp,posYp,posZp,x,y,z)
                                # vertexPixelValue = getPixelValueIntersectionPointPython(posXp,posYp,posZp,x,y,z)
                                # print("Intersecting point :", intersectionPoint," pixel Value :",vertexPixelValue)
                                # print(vertexPixelValue2)
                                # print(vertexPixelValue)
                                
                                # intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                #                     math.floor(vertexPixelValue[0]), math.floor(vertexPixelValue[1]),x,y,z,mp,mq]
                                
                                #planeEdgeIntersectionPython(5,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq)
                                x = intersectionPoint[0]
                                y = intersectionPoint[1]
                                z = intersectionPoint[2]
                                # vertexPixelValue2 = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
                                vertexPixelValue2 = getVertexPixelValueIntersectZ3(x,y,z)
                                xpixel = vertexPixelValue2[0]
                                ypixel = vertexPixelValue2[1]
                                # print(xpixel,ypixel)
                                if(xpixel<0 or ypixel <0):
                                        continue
                                intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                                            xpixel, ypixel,x,y,z,mp,mq]
                                
                                
                                
                                
                                
                                
                                # intersectingData = [j, insideVertex,outsideVertex,currentIntersectingPlane,\
                                #                         vertexPixelValue[0], vertexPixelValue[1],x,y,z,mp,mq]
                                
                                #print("intersecting data :",intersectingData)
                                intersectingEdgeDataToPPL.append(intersectingData)
                                numberOfIntersectingEdges += 1
                                currImage.append([math.floor(float(vertexPixelValue2[0])),\
                                        math.floor(float(vertexPixelValue2[1]))])
                            
                
                    
                
       
        
        currGroupName ="dummyGroup"
        currImageName ="singlePosImage"
        
        currImageSetConStringPolyhedra = pyparma_posInvRegion32.computeRegion(currGroupName,posZp,numberOfFullyInsideVertices,insideVertexDetailsToPPL,\
            numberOfIntersectingEdges,intersectingEdgeDataToPPL,posXp,posYp,posZp,m[xp0],m[yp0],m[zp0], outcodeP0,currImageName,currRegionPPL)
        
        
        
        return currImageSetConStringPolyhedra
        
        # currImageSetConString = currImageSetConString.replace("x0","xp0")
        # currImageSetConString = currImageSetConString.replace("x1","yp0")
        # currImageSetConString = currImageSetConString.replace("x2","zp0")
        # currImageSetConString = currImageSetConString.replace(" = ","==")
        # currImageSetConString = currImageSetConString.replace("Constraint_System {"," ")
        # currImageSetConString = currImageSetConString.replace("}"," ")
        
        # print("\nAfter replacing \n")
        # print(currImageSetConString)
        
      