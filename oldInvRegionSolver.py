






import environment

from z3 import *


# vertices = environment.vertices
# nvertices = environment.nvertices
# edges = environment.tedges


vertices = [50 , 6 , 190 ,  48 , 1 , 190 ,  52 , 1 , 190 ]
nvertices =[ 0 , 1 , 2  ]
edges = [0 , 1 , 1 , 2 , 2 , 0]

xp0, yp0, zp0 = Reals('xp0 yp0 zp0')

t = environment.t
b = environment.b
l = environment.l
r = environment.r
n = environment.n
f = environment.f

# OpenGL perspective projection matrix
mProj = [
        [2 * n / (r - l), 0, 0, 0],
        [0,2 * n / (t - b),0,0],
        [(r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n), -1 ],
        [0,0,-2 * f * n / (f - n),0]
    ]


def computeOutcodeAtPos(outcodeP0, inx, iny,inz):
    
    outcode = 0   
    outx   = inx * mProj[0][0] + iny * mProj[1][0] + inz * mProj[2][0] +  mProj[3][0]
    outy   = inx * mProj[0][1] + iny * mProj[1][1] + inz * mProj[2][1] +  mProj[3][1] 
    outz   = inx * mProj[0][2] + iny * mProj[1][2] + inz * mProj[2][2] +  mProj[3][2] 
    w = inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3] 

    i =0
      
    
    # outValues[i*4+0] = outx
    # outValues[i*4+1] = outy
    # outValues[i*4+2] = outz
    # outValues[i*4+3] = w
    
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

def getVertexPixelValueIntersectZ3_2(x,y,z):
    s3 = Solver()
    set_param('parallel.enable', True)
    set_option(rational_to_decimal=False)
    # set_option(precision=20)
    set_param('parallel.enable', True)
    s3.set("sat.local_search_threads", 28)
    s3.set("sat.threads", 28)
    s3.set("sat.threads", 28)
    s3.check()
    
    a,b = Reals('a b')
    cons1 = ( (((-68.39567*(x ) )/ (z) )+ 24.5 ) == a )
    cons2 =  ( (((68.39567*(y ) )/ (z) )+ 24.5 ) == b )
    
    s3.add(simplify(And(cons1,cons2)))
    p = [0,0]
    # print(s3.check())
    # print(cons1)
    # print(cons2)
    if s3.check() == sat:
        m2 = s3.model()
        # print("etVertexPixelValueIntersectZ3: model from solver :",m2)
        a = str(eval("m2[a].numerator_as_long()/m2[a].denominator_as_long()"))
        b = str(eval("m2[b].numerator_as_long()/m2[b].denominator_as_long()"))
        print(a,b)    
        p[0] = math.floor(float(a))
        p[1] = math.floor(float(b))

        # print(p)
            
        return p
    else:
        print("no sat image")
        return p


def getVertexPixelValueZ3_2(m,x,y,z):
    s3 = Solver()
    set_param('parallel.enable', True)
    set_option(rational_to_decimal=False)
    # set_option(precision=20)
    set_param('parallel.enable', True)
    s3.set("sat.local_search_threads", 28)
    s3.set("sat.threads", 28)
    s3.set("sat.threads", 28)
    s3.check()
    
    a,b = Reals('a b')
    cons1 = ( (((-68.39567*(x - m[xp0] ) )/ (z - m[zp0]) )+ 24.5 ) == a )
    cons2 =  ( (((68.39567*(y  - m[yp0]) )/ (z - m[zp0]) )+ 24.5 ) == b )
    
    # xp =50
    # yp =4.5
    # zp = 250
    # print("m =", m)
    # cons1 = "( (((-68.39567*(x -xp) )/ (z -zp) )+ 24.5 ) == a )"
    # cons2 =  "( (((68.39567*(y -yp) )/ (z -zp) )+ 24.5 ) == b )"

    # print(cons1)
    # print(cons2)
    s3.add(simplify(And(cons1,cons2)))
    p = [0,0]
    if s3.check() == sat:
        m2= s3.model()
        #print("model from solver :",m)
        a = str(eval("m2[a].numerator_as_long()/m2[a].denominator_as_long()"))
        b = str(eval("m2[b].numerator_as_long()/m2[b].denominator_as_long()"))
        print(a,b)    
        p[0] = math.floor(float(a))
        p[1] = math.floor(float(b))
            
        return p
    else:
        print("no sat image")
        exit()  
    



# plane0 = top, plane1 = left, plane2=bottom, plane3= right
def planeEdgeIntersection(plane,insideVertex, outsideVertex,posXp,posYp,posZp,intersectionPoint,vertexPixelValue2,mp,mq):
    
    # print("\n\n")
    s1 = Solver()
    set_param('parallel.enable', True)
    set_option(rational_to_decimal=False)
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
        # print(f"intersecting with plane {plane} using linear combination;")
        m = s1.model()
        
        insideFraction = eval("m[p].numerator_as_long()/m[p].denominator_as_long() " )
        outsideFraction = eval("m[q].numerator_as_long()/m[q].denominator_as_long() " )
        
        vertexPixelValue2 = getVertexPixelValueIntersectZ3_2(m[p]*xv0+m[q]*xv1,\
                                                           m[p]*yv0+m[q]*yv1,\
                                                           m[p]*zv0+m[q]*zv1)  
        # print(vertexPixelValue2)
                                
        # print("p, q : ",m[p],m[q])
        # print("vertex 0 :",xv0,yv0,zv0)
        # print("vertex 1 :",xv1,yv1,zv1)
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
        return 1, vertexPixelValue2
    elif result == unsat:
        # print("not intersecting with the left plane")
        return 0  , vertexPixelValue2 
    else:
        print("timeout occured")
        return 0, vertexPixelValue2



# plane0 = top, plane1 = left, plane2=bottom, plane3= right
def planeEdgeIntersectionConstraint(plane,insideVertex, outsideVertex, currXpixel, currYpixel, currVarsIndex,
                                    pVars, aVars, bVars,cVars,dVars):
    
                                                            
    #print("outside vertex :", outsideVertex)
    #print(posXp,posYp,posZp)
    
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


           
        consE = (((-68.39567*(((pVars[currVarsIndex] * (vertices[insideVertex*3+0] - xp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+0] - xp0))) )/ (((pVars[currVarsIndex] * (vertices[insideVertex*3+2] - zp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+2] - zp0))) )+ 24.5 ) >= currXpixel) )
        consF = (((-68.39567*(((pVars[currVarsIndex] * (vertices[insideVertex*3+0] - xp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+0] - xp0))) )/ (((pVars[currVarsIndex] * (vertices[insideVertex*3+2] - zp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+2] - zp0))) )+ 24.5 ) < currXpixel+1) )
        
        
    
    if(plane == 1):
        # print("checking intersection with the bottom plane")
           

        # plane0_v0 = [-0.35820895522388063,-0.35820895522388063,-1]
        # plane0_v1 = [0.35820895522388063,-0.35820895522388063,-1]
        # plane0_v2 = [-358.20895522388063,-358.20895522388063,-1000]    
        # plane0_v3 = [358.20895522388063,-358.20895522388063,-1000] 

        plane0_v0 = [xp0-0.35820895522388063,yp0-0.35820895522388063,zp0-1]
        plane0_v1 = [xp0+0.35820895522388063,yp0-0.35820895522388063,zp0-1]
        plane0_v2 = [xp0-358.20895522388063,yp0-358.20895522388063,zp0-1000]    
        plane0_v3 = [xp0+358.20895522388063,yp0-358.20895522388063,zp0-1000] 
        
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

        consE = (((-68.39567*(((pVars[currVarsIndex] * (vertices[insideVertex*3+0] - xp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+0] - xp0))) )/ (((pVars[currVarsIndex] * (vertices[insideVertex*3+2] - zp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+2] - zp0))) )+ 24.5 ) >= currXpixel) )
        consF = (((-68.39567*(((pVars[currVarsIndex] * (vertices[insideVertex*3+0] - xp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+0] - xp0))) )/ (((pVars[currVarsIndex] * (vertices[insideVertex*3+2] - zp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+2] - zp0))) )+ 24.5 ) < currXpixel+1) )
       


      
    
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

        consE = (((-68.39567*(((pVars[currVarsIndex] * (vertices[insideVertex*3+1] - yp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+1] - yp0))) )/ (((pVars[currVarsIndex] * (vertices[insideVertex*3+2] - zp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+2] - zp0))) )+ 24.5 ) >= currYpixel) )
        consF = (((-68.39567*(((pVars[currVarsIndex] * (vertices[insideVertex*3+1] - yp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+1] - yp0))) )/ (((pVars[currVarsIndex] * (vertices[insideVertex*3+2] - zp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+2] - zp0))) )+ 24.5 ) < currYpixel+1) )
    
           
    
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

        consE = (((-68.39567*(((pVars[currVarsIndex] * (vertices[insideVertex*3+1] - yp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+1] - yp0))) )/ (((pVars[currVarsIndex] * (vertices[insideVertex*3+2] - zp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+2] - zp0))) )+ 24.5 ) >= currYpixel) )
        consF = (((-68.39567*(((pVars[currVarsIndex] * (vertices[insideVertex*3+1] - yp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+1] - yp0))) )/ (((pVars[currVarsIndex] * (vertices[insideVertex*3+2] - zp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+2] - zp0))) )+ 24.5 ) < currYpixel+1) )
       


                         
    
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


        consE = (((-68.39567*(((pVars[currVarsIndex] * (vertices[insideVertex*3+0] - xp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+0] - xp0))) )/ (((pVars[currVarsIndex] * (vertices[insideVertex*3+2] - zp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+2] - zp0))) )+ 24.5 ) >= currXpixel) )
        consF = (((-68.39567*(((pVars[currVarsIndex] * (vertices[insideVertex*3+0] - xp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+0] - xp0))) )/ (((pVars[currVarsIndex] * (vertices[insideVertex*3+2] - zp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+2] - zp0))) )+ 24.5 ) < currXpixel+1) )
        consE2 = (((-68.39567*(((pVars[currVarsIndex] * (vertices[insideVertex*3+1] - yp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+1] - yp0))) )/ (((pVars[currVarsIndex] * (vertices[insideVertex*3+2] - zp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+2] - zp0))) )+ 24.5 ) >= currYpixel) )
        consF2 = (((-68.39567*(((pVars[currVarsIndex] * (vertices[insideVertex*3+1] - yp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+1] - yp0))) )/ (((pVars[currVarsIndex] * (vertices[insideVertex*3+2] - zp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+2] - zp0))) )+ 24.5 ) < currYpixel+1) )
       



    

    consA = (aVars[currVarsIndex]*x0+bVars[currVarsIndex]*x1+cVars[currVarsIndex]*x2+dVars[currVarsIndex]*x3) == (pVars[currVarsIndex] * (vertices[insideVertex*3+0] - xp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+0] - xp0))
    consB = (aVars[currVarsIndex]*y0+bVars[currVarsIndex]*y1+cVars[currVarsIndex]*y2+dVars[currVarsIndex]*y3) == (pVars[currVarsIndex] * (vertices[insideVertex*3+1] - yp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+1] - yp0))
    consC = (aVars[currVarsIndex]*z0+bVars[currVarsIndex]*z1+cVars[currVarsIndex]*z2+dVars[currVarsIndex]*z3) == (pVars[currVarsIndex] * (vertices[insideVertex*3+2] - zp0) + (1- pVars[currVarsIndex])* (vertices[outsideVertex*3+2] - zp0))

    consD = And(aVars[currVarsIndex] >=0, bVars[currVarsIndex] >=0, cVars[currVarsIndex] >=0, dVars[currVarsIndex] >=0)
    consG = And(aVars[currVarsIndex] + bVars[currVarsIndex] + cVars[currVarsIndex] + dVars[currVarsIndex] == 1)

    
    if plane == 5:
        consReturn = Exists([pVars[currVarsIndex], aVars[currVarsIndex], bVars[currVarsIndex], cVars[currVarsIndex], dVars[currVarsIndex]] ,And(consA, consB, consC, consD, consG, consE, consF, consE2, consF2))

        qe_formula = Tactic('qe').apply(consReturn)


    else:
        consReturn = Exists([pVars[currVarsIndex], aVars[currVarsIndex], bVars[currVarsIndex], cVars[currVarsIndex], dVars[currVarsIndex]] ,And(consA, consB, consC, consD, consG, consE, consF))

        # print(consReturn)
        # s1 = Solver()

        tactic = Then(Tactic('simplify'), Tactic('solve-eqs'), Tactic('propagate-ineqs'), Tactic('qe'))

        # s1.add(consReturn)
        # qe_formula = Tactic('qe2').apply(consReturn)
        qe_formula = tactic(consReturn)

        # Print the result
        # print(qe_formula)



        
    return qe_formula



        
    
    
  




def computeRegion3(numberOfFullyInsideVertices, insideVertexDetailsToPPL, 
                            numberOfIntersectingEdges, intersectingEdgeDataToPPL,
                            currVarsIndex, pVars, aVars, bVars,cVars,dVars):
    

    currCons = And(True)
    for i in range(0,numberOfFullyInsideVertices):
        print("fully inside vertex")
        print(insideVertexDetailsToPPL[i])

        x = vertices[insideVertexDetailsToPPL[i][0]*3+0]
        y = vertices[insideVertexDetailsToPPL[i][0]*3+1]
        z = vertices[insideVertexDetailsToPPL[i][0]*3+2]

        print("x,y,z : ",x,y,z)

        cons1 = ( (((-68.39567*(x -xp0) )/ (z -zp0) )+ 24.5 ) >= insideVertexDetailsToPPL[i][1] )
        cons2 =  ( (((68.39567*(y -yp0) )/ (z -zp0) )+ 24.5 ) >= insideVertexDetailsToPPL[i][2] )
        cons3 = ( (((-68.39567*(x -xp0) )/ (z -zp0) )+ 24.5 ) < insideVertexDetailsToPPL[i][1]+1 )
        cons4 =  ( (((68.39567*(y -yp0) )/ (z -zp0) )+ 24.5 ) < insideVertexDetailsToPPL[i][2]+1 )

        currCons = And(currCons, cons1, cons2, cons3, cons4)
    
    print("currCons : ",currCons)  
    print("currCons : ",simplify(currCons)   )

    for i in range(0,numberOfIntersectingEdges):
        print("\n=====\nintersecting edge")
        print(intersectingEdgeDataToPPL[i])

        inVertex = intersectingEdgeDataToPPL[i][1]
        outVertex = intersectingEdgeDataToPPL[i][2]
        intersectingPlane = intersectingEdgeDataToPPL[i][3]

        currXpixel = intersectingEdgeDataToPPL[i][4]
        currYpixel = intersectingEdgeDataToPPL[i][5]

        edgeIntersectCons = planeEdgeIntersectionConstraint(intersectingPlane,inVertex, outVertex,
                                                            currXpixel, currYpixel, currVarsIndex,
                                                            pVars, aVars, bVars,cVars,dVars)

        print("\n\nedge intersect cons : ",edgeIntersectCons[0][0])
        currCons = And(currCons, edgeIntersectCons[0][0])
        currVarsIndex += 1
    

    return currCons, currVarsIndex

    





def invUsingConstraints(posXp, posYp, posZp, currTriangle,m,
                        currVarsIndex, pVars, aVars, bVars,cVars,dVars):

    vertex0 = nvertices[currTriangle*3+0]
    vertex1 = nvertices[currTriangle*3+1]
    vertex2 = nvertices[currTriangle*3+2]
    currTriangleVertices = [vertex0, vertex1,vertex2]
    
    
   


    v0Vertex = [vertices[currTriangleVertices[0]*3+0], vertices[currTriangleVertices[0]*3+1],vertices[currTriangleVertices[0]*3+2] ]
    v1Vertex = [vertices[currTriangleVertices[1]*3+0], vertices[currTriangleVertices[1]*3+1],vertices[currTriangleVertices[1]*3+2] ]
    v2Vertex = [vertices[currTriangleVertices[2]*3+0], vertices[currTriangleVertices[2]*3+1],vertices[currTriangleVertices[2]*3+2] ]


    outcodeP0 = [0,0,0,0,0,0]
    outcodeP1 = [0,0,0,0,0,0]
    outcodeP2 = [0,0,0,0,0,0]
    computeOutcodeAtPos(outcodeP0, v0Vertex[0] -posXp ,v0Vertex[1]-posYp,v0Vertex[2]-posZp )
    computeOutcodeAtPos(outcodeP1, v1Vertex[0] -posXp ,v1Vertex[1]-posYp,v1Vertex[2]-posZp )
    computeOutcodeAtPos(outcodeP2, v2Vertex[0] -posXp ,v2Vertex[1]-posYp,v2Vertex[2]-posZp )

    currEdges = [edges[currTriangle*6+0], edges[currTriangle*6+1],edges[currTriangle*6+2], edges[currTriangle*6+3], 
                 edges[currTriangle*6+4], edges[currTriangle*6+5]]
    

    # 0:fully inside,  1: intersect, 2: fully outside
    edgesOutInInter =[0]*3

    pixelValueComputed = [0]*3

    insideVertexDetailsToPPL = [] #store vertex index number,xpixel,ypixel
    numberOfFullyInsideVertices = 0
    numberOfIntersectingEdges = 0
    intersectingEdgeDataToPPL = []
    
    
    print("classify edges inside, outside, intersec :start")
    for j in range(0,3):
        print("\n\nedge : ",j)
        edge_v0 = currEdges[j*2+0]
        edge_v1 = currEdges[j*2+1]

        edge_v0IndexInCurrTriangleVertices = 0
        edge_v1IndexInCurrTriangleVertices = 0
        print(edge_v0,edge_v1)



        currOutCode0 = []
        currOutCode1 = []

        if edge_v0 == currTriangleVertices[0]:
            currOutCode0 = outcodeP0
            edge_v0IndexInCurrTriangleVertices = 0
        elif edge_v0 == currTriangleVertices[1]:
            currOutCode0 = outcodeP1
            edge_v0IndexInCurrTriangleVertices = 1
        else:
            currOutCode0 = outcodeP2
            edge_v0IndexInCurrTriangleVertices = 2

        if edge_v1 == currTriangleVertices[0]:
            currOutCode1 = outcodeP0
            edge_v1IndexInCurrTriangleVertices = 0
        elif edge_v1 == currTriangleVertices[1]:
            currOutCode1 = outcodeP1
            edge_v1IndexInCurrTriangleVertices = 1
        else:
            currOutCode1 = outcodeP2
            edge_v1IndexInCurrTriangleVertices = 2


       
        # check whether the edge intersect, fullyoutside, or inside
        if (all(element == 0 for element in currOutCode0)  and all(element == 0 for element in currOutCode1) ):
            edgesOutInInter[j] = 0
        elif( (currOutCode0[0] == 1 and currOutCode1[0] == 1) or\
                (currOutCode0[1] == 1 and currOutCode1[1] == 1) or \
                (currOutCode0[2] == 1 and currOutCode1[2] == 1) or \
                (currOutCode0[3] == 1 and currOutCode1[3] == 1) or \
                (currOutCode0[4] == 1 and currOutCode1[4] == 1) or\
                (currOutCode0[5] == 1 and currOutCode1[5] == 1)):
            edgesOutInInter[j] = 2
            print("fully outside edge")
        else:
            edgesOutInInter[j] = 1

        
        print("edge ",j," : ",edgesOutInInter[j])

        if(edgesOutInInter[j] == 0):
            # fully inside
            print("fully inside edge")
            # edge_v0 = tedges[j*2+0]
            # edge_v1 = tedges[j*2+1] 
            if(pixelValueComputed[edge_v0IndexInCurrTriangleVertices] == 0):
                pixelValueComputed[edge_v0IndexInCurrTriangleVertices] = 1
                x = vertices[edge_v0*3+0]
                y = vertices[edge_v0*3+1]
                z = vertices[edge_v0*3+2]
                print("x,y,z : ",x,y,z)
                vertexPixelValue = getVertexPixelValueZ3_2(m,x,y,z)
                # # add the vertex details to PPL ds
                tempPixelData = [edge_v0, vertexPixelValue[0],vertexPixelValue[1]]
                insideVertexDetailsToPPL.append(tempPixelData)
                numberOfFullyInsideVertices+=1
                print("vertex pixel value : ",vertexPixelValue)
                
            
            if(pixelValueComputed[edge_v1IndexInCurrTriangleVertices] == 0):
                pixelValueComputed[edge_v1IndexInCurrTriangleVertices] = 1
                x = vertices[edge_v1*3+0]
                y = vertices[edge_v1*3+1]
                z = vertices[edge_v1*3+2]
                print("x,y,z : ",x,y,z)
                vertexPixelValue = getVertexPixelValueZ3_2(m,x,y,z)
                # # add the vertex details to PPL ds
                tempPixelData = [edge_v1, vertexPixelValue[0],vertexPixelValue[1]]
                insideVertexDetailsToPPL.append(tempPixelData)
                numberOfFullyInsideVertices+=1
                print("vertex pixel value : ",vertexPixelValue)
        
        elif(edgesOutInInter[j] == 1):
            # intersecting edge
            print("intersecting edge")

            
            sumOfOutcode_v0 = currOutCode0[0]+currOutCode0[1] +currOutCode0[2]+ currOutCode0[3]+currOutCode0[4] +currOutCode0[5]
            sumOfOutcode_v1 = currOutCode1[0]+currOutCode1[1] +currOutCode1[2]+ currOutCode1[3]+currOutCode1[4] +currOutCode1[5]



           

            outsideVertex = edge_v1IndexInCurrTriangleVertices
            insideVertex  = edge_v0IndexInCurrTriangleVertices
            
           
            
            bothVerticesOutside = 0
            if ( sumOfOutcode_v0== 0 and sumOfOutcode_v1 >0):
                insideVertex  = edge_v0IndexInCurrTriangleVertices
                outsideVertex = edge_v1IndexInCurrTriangleVertices
            elif(sumOfOutcode_v0 > 0 and sumOfOutcode_v1 == 0):
                insideVertex  = edge_v1IndexInCurrTriangleVertices
                outsideVertex = edge_v0IndexInCurrTriangleVertices
            else:
                bothVerticesOutside = 1
            
            print("bothVerticesOutside : ",bothVerticesOutside)
            print("insideVertex : ",insideVertex)
            print("outsideVertex : ",outsideVertex)

            if(bothVerticesOutside == 0 ):
                print("one vertex is inside")
                
                if(pixelValueComputed[insideVertex] == 0):
                    pixelValueComputed[insideVertex] = 1
                    x = vertices[currTriangleVertices[insideVertex]*3+0]
                    y = vertices[currTriangleVertices[insideVertex]*3+1]
                    z = vertices[currTriangleVertices[insideVertex]*3+2]
                    print("x,y,z : ",x,y,z)
                    print(m)
                    vertexPixelValue = getVertexPixelValueZ3_2(m,x,y,z)
                    tempPixelData = [currTriangleVertices[insideVertex], vertexPixelValue[0],vertexPixelValue[1]]
                    insideVertexDetailsToPPL.append(tempPixelData)
                    numberOfFullyInsideVertices+=1
                    print("vertex pixel value : ",vertexPixelValue)
            

            intersectionPoint = [0,0,0,0]
            vertexPixelValue2 = [0,0]
            mp =0
            mq =0
            if(currOutCode0[0] != 0 or currOutCode1[0] != 0):
                # top
                print("top")

                result, vertexPixelValue2 = planeEdgeIntersection(0,currTriangleVertices[insideVertex], currTriangleVertices[outsideVertex],
                                         m[xp0],m[yp0],m[zp0],intersectionPoint, vertexPixelValue2,mp,mq) 
                if(result == 1):
                    # print("intersecting with top plane")
                    currentIntersectingPlane = 0
                    # print(intersectionPoint)
                    x = intersectionPoint[0]
                    y = intersectionPoint[1]
                    z = intersectionPoint[2]

                    # print("x,y,z : ",x,y,z)
                    print("vertex pixel value : ",vertexPixelValue2)


                    intersectingData = [j, currTriangleVertices[insideVertex],currTriangleVertices[outsideVertex],
                                                currentIntersectingPlane,
                                                vertexPixelValue2[0], vertexPixelValue2[1],x,y,z,mp,mq]
                    #print("intersecting data :",intersectingData)
                    intersectingEdgeDataToPPL.append(intersectingData)
                    numberOfIntersectingEdges += 1
            
            if(currOutCode0[1] != 0 or currOutCode1[1] != 0):
                # bottom
                print("bottom")
                # print(currOutCode0[3],currOutCode1[3])

                result, vertexPixelValue2 = planeEdgeIntersection(1,currTriangleVertices[insideVertex], currTriangleVertices[outsideVertex],
                                         m[xp0],m[yp0],m[zp0],intersectionPoint, vertexPixelValue2,mp,mq) 
                if(result == 1):
                    # print("intersecting with top plane")
                    currentIntersectingPlane = 1
                    # print(intersectionPoint)
                    x = intersectionPoint[0]
                    y = intersectionPoint[1]
                    z = intersectionPoint[2]

                    # print("x,y,z : ",x,y,z)
                    print("vertex pixel value : ",vertexPixelValue2)


                    intersectingData = [j, currTriangleVertices[insideVertex],currTriangleVertices[outsideVertex],
                                                currentIntersectingPlane,
                                                vertexPixelValue2[0], vertexPixelValue2[1],x,y,z,mp,mq]
                    #print("intersecting data :",intersectingData)
                    intersectingEdgeDataToPPL.append(intersectingData)
                    numberOfIntersectingEdges += 1
        
            if(currOutCode0[2] != 0 or currOutCode1[2] != 0):
                # right
                print("right")

                result, vertexPixelValue2 = planeEdgeIntersection(2,currTriangleVertices[insideVertex], currTriangleVertices[outsideVertex],
                                         m[xp0],m[yp0],m[zp0],intersectionPoint, vertexPixelValue2,mp,mq) 
                if(result == 1):
                    # print("intersecting with top plane")
                    currentIntersectingPlane = 2
                    # print(intersectionPoint)
                    x = intersectionPoint[0]
                    y = intersectionPoint[1]
                    z = intersectionPoint[2]

                    # print("x,y,z : ",x,y,z)
                    print("vertex pixel value : ",vertexPixelValue2)


                    intersectingData = [j, currTriangleVertices[insideVertex],currTriangleVertices[outsideVertex],
                                                currentIntersectingPlane,
                                                vertexPixelValue2[0], vertexPixelValue2[1],x,y,z,mp,mq]
                    #print("intersecting data :",intersectingData)
                    intersectingEdgeDataToPPL.append(intersectingData)
                    numberOfIntersectingEdges += 1
            
            if(currOutCode0[3] != 0 or currOutCode1[3] != 0):
                # left
                print("left")
                
                result, vertexPixelValue2 = planeEdgeIntersection(3,currTriangleVertices[insideVertex], currTriangleVertices[outsideVertex],
                                         m[xp0],m[yp0],m[zp0],intersectionPoint, vertexPixelValue2,mp,mq) 
                if(result == 1):
                    # print("intersecting with top plane")
                    currentIntersectingPlane = 3
                    # print(intersectionPoint)
                    x = intersectionPoint[0]
                    y = intersectionPoint[1]
                    z = intersectionPoint[2]

                    # print("x,y,z : ",x,y,z)
                    print("vertex pixel value : ",vertexPixelValue2)


                    intersectingData = [j, currTriangleVertices[insideVertex],currTriangleVertices[outsideVertex],
                                                currentIntersectingPlane,
                                                vertexPixelValue2[0], vertexPixelValue2[1],x,y,z,mp,mq]
                    #print("intersecting data :",intersectingData)
                    intersectingEdgeDataToPPL.append(intersectingData)
                    numberOfIntersectingEdges += 1
            
            
            
            if(currOutCode0[5] != 0 or currOutCode1[5] != 0):
                # near
                print("near")

                result, vertexPixelValue2 = planeEdgeIntersection(5,currTriangleVertices[insideVertex], currTriangleVertices[outsideVertex],
                                         m[xp0],m[yp0],m[zp0],intersectionPoint, vertexPixelValue2,mp,mq) 
                if(result == 1):
                    # print("intersecting with top plane")
                    currentIntersectingPlane = 5
                    # print(intersectionPoint)
                    x = intersectionPoint[0]
                    y = intersectionPoint[1]
                    z = intersectionPoint[2]

                    # print("x,y,z : ",x,y,z)
                    print("vertex pixel value : ",vertexPixelValue2)


                    intersectingData = [j, currTriangleVertices[insideVertex],currTriangleVertices[outsideVertex],
                                                currentIntersectingPlane,
                                                vertexPixelValue2[0], vertexPixelValue2[1],x,y,z,mp,mq]
                    #print("intersecting data :",intersectingData)
                    intersectingEdgeDataToPPL.append(intersectingData)
                    numberOfIntersectingEdges += 1

    print("number of fully inside vertices : ",numberOfFullyInsideVertices)
    print("fully inside data : ",insideVertexDetailsToPPL)
    print("number of intersecting edges : ",numberOfIntersectingEdges)
    print("intersecting edge data : ",intersectingEdgeDataToPPL)

    # numberOfIntersectingEdges = 1

    regCons, currVarsIndex = computeRegion3(numberOfFullyInsideVertices, insideVertexDetailsToPPL, 
                            numberOfIntersectingEdges, intersectingEdgeDataToPPL,
                            currVarsIndex, pVars, aVars, bVars,cVars,dVars)
    

    return regCons, currVarsIndex  




        
               


no_of_vars = 100

pVars = []
aVars = []
bVars = []
cVars = []
dVars = []


for iii in range(no_of_vars):
    pVars += [Real('p%d' % iii)]
    aVars += [Real('a%d' % iii)]
    bVars += [Real('b%d' % iii)]
    cVars += [Real('c%d' % iii)]
    dVars += [Real('d%d' % iii)]
    
# currVarsIndex = 0




ss = Solver()

set_param('parallel.enable', True)
set_option(rational_to_decimal=True)
# set_option(precision=1000)
set_param('parallel.enable', True)
ss.set("sat.local_search_threads", 50)
ss.set("sat.threads", 50)
# s2.set("timeout",2000)    
set_option(max_args=10000000, max_lines=1000000, max_depth=10000000, max_visited=1000000)

ss.add(And(xp0>=50,100*xp0<=5001,10*yp0>=45,100*yp0<=451, 10*zp0>=1985,100*zp0<=19851))

initRegionCons = And(xp0>=50,100*xp0<=5001,10*yp0>=45,100*yp0<=451, 10*zp0>=1985,100*zp0<=19851)
# # print(ss.check())
# m = ss.model()
# posXp = (eval("m[xp0].numerator_as_long()/m[xp0].denominator_as_long()"))
# posYp = (eval("m[yp0].numerator_as_long()/m[yp0].denominator_as_long()"))
# posZp = (eval("m[zp0].numerator_as_long()/m[zp0].denominator_as_long()"))

# print(posXp, posYp, posZp)
currTriangle = 0
# regCons, currVarsIndex   = invUsingConstraints(posXp, posYp, posZp, currTriangle,m, 
#                              currVarsIndex, pVars, aVars, bVars,cVars,dVars)

# print("\n\n\n\n--------------------\ndone")
# print(simplify(regCons), currVarsIndex)

# ss.add(Not(regCons))
# print(ss.check())
# print(m)
# print(ss.model())

# ss.add(Not(regCons))
# print(ss.check())

count = 0

from time import sleep
while(ss.check()==sat):
    m = ss.model()
    posXp = (eval("m[xp0].numerator_as_long()/m[xp0].denominator_as_long()"))
    posYp = (eval("m[yp0].numerator_as_long()/m[yp0].denominator_as_long()"))
    posZp = (eval("m[zp0].numerator_as_long()/m[zp0].denominator_as_long()"))

    print(posXp, posYp, posZp)
    currTriangle = 0
    regCons, currVarsIndex   = invUsingConstraints(posXp, posYp, posZp, currTriangle,m, 
                                currVarsIndex, pVars, aVars, bVars,cVars,dVars)

    print("\n\n\n\n--------------------\ndone")
    print(simplify(regCons), currVarsIndex)

    currVarsIndex += 1

    sss = Solver()
    sss.add(regCons)
    sss.add(initRegionCons)
    # sss.add(And(xp0 == m[xp0], yp0 == m[yp0], zp0 == m[zp0]))
    sss.add(And(xp0 == posXp, yp0 == posYp, zp0 == posZp))
    print(sss.check())
    print(sss.model())

    ss.add(Not(regCons))
    ss.add(Or(xp0 != m[xp0], yp0 != m[yp0], zp0 != m[zp0]))
    count +=1
    print("currnet number of regions = ",count)
    print(posXp, posYp, posZp)
    sleep(5)


print("Program Finished")







