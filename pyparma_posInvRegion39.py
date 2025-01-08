from pyparma import *
from z3 import *
import environment
import math
from time import sleep
import numpy as np
from fractions import Fraction


def mygcd(a, b):
    
    if (a == 0):
        return b

    return mygcd(b % a, a)


def lcm(a, b):
    # #print("computing lcm")
    # return a*b //mygcd(a,b)
    # print(a, b)
    # a = np.asarray(a, dtype='float64')
    # b = np.asarray(b, dtype='float64')
    # print(a,b)
    try:
        return np.lcm(a, b)
    except:
        # print("second method")
        # sleep(4)
        return a*b // mygcd(a, b)


vertices = environment.vertices
xp0 = Variable(0)
yp0 = Variable(1)
zp0 = Variable(2)
vertices = environment.vertices
numOfVertices = environment.numOfVertices
tedges = environment.tedges
numOftedges = environment.numOfEdges
# intiFrusCons = environment.intiFrusCons
initCubeCon = environment.initCubeCon
# x0 = environment.x0
# x1 = environment.x1
# y0 = environment.y0
# y1 = environment.y1
# zmin = environment.z0
# zmax = environment.z1
canvasWidth = environment.canvasWidth
canvasHeight = environment.canvasHeight
focalLength = environment.focalLength
t = environment.t
b = environment.b
l = environment.l
r = environment.r
n = environment.n
f = environment.f

unsatFlag = 0
def addAvertexPixelConstraintIntersect(currZP, x,y,z, pixelX, pixelY, pd,planeId):
   
    if (isinstance(x, float)):
        print("float x")
        xf = str(Fraction(x).limit_denominator())
        xl = xf.split('/')
        px = int(xl[0])
        # qx = int(xl[1])
        if(len(xl) == 2):
            qx = int(xl[1])
        else:
            qx = 1

    else:
        px = x
        qx = 1
    if (isinstance(y, float)):
        print("float y")
        yf = str(Fraction(y).limit_denominator())
        yl = yf.split('/')
        py = int(yl[0])
        # qy = int(yl[1])
        if(len(yl) == 2):
            qy = int(yl[1])
        else:
            qy = 1
    else:
        py = y
        qy = 1
    if (isinstance(z, float)):
        print("float z")
        zf = str(Fraction(z).limit_denominator())
        zl = zf.split('/')
        pz = int(zl[0])
        # qz = int(zl[1])
        if(len(zl) == 2):
            qz = int(zl[1])
        else:
            qz = 1
    else:
        pz = z
        qz = 1
    # xf = str(Fraction(x).limit_denominator())
    # yf = str(Fraction(y).limit_denominator())
    # zf = str(Fraction(z).limit_denominator())
    # xl = xf.split('/')
    # yl = yf.split('/')
    # zl = zf.split('/')
    # px = int(xl[0])
    # qx = int(xl[1])
    # py = int(yl[0])
    # qy = int(yl[1])
    # pz = int(zl[0])
    # qz = int(zl[1])

    # print("adding inside vertex cons vertexnumber : ",vertexNumber,pixelX,pixelY)
    # print(x,y,z)
    # print(px, qx, py, qy, pz, qz)

    if(z - currZP < 0):
        print("z - currZP <0")
        
        # if (planeId == 0):
        print("top")
        pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
                            (pz - qz*zp0)*1)*pow(10,5) ) <= ((pixelX)*qx*(pz - qz*zp0)*1)*pow(10,6) )
        
        
        pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
                        (pz - qz*zp0)*1)*pow(10,5) ) > (((pixelX+1))*qx*(pz - qz*zp0)*1)*pow(10,6) )

        pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
                            (pz - qz*zp0)*1)*pow(10,5) ) <= ((pixelY)*qy*(pz - qz*zp0)*1)*pow(10,6) )
        pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
                            (pz - qz*zp0)*1)*pow(10,5) ) > (((pixelY+1))*qy*(pz - qz*zp0)*1)*pow(10,6) )
        
        # if (planeId == 1):
        #     print("bottom")
        #     pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
        #                         (pz - qz*zp0)*1)*pow(10,5) ) <= ((pixelX)*qx*(pz - qz*zp0)*1)*pow(10,6) )
            
            
        #     pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
        #                     (pz - qz*zp0)*1)*pow(10,5) ) > (((pixelX+1))*qx*(pz - qz*zp0)*1)*pow(10,6) )

        #     pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
        #                       (pz - qz*zp0)*1)*pow(10,5) ) <= ((pixelY)*qy*(pz - qz*zp0)*1)*pow(10,6) )
        #     pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
        #                       (pz - qz*zp0)*1)*pow(10,5) ) > (((pixelY+1))*qy*(pz - qz*zp0)*1)*pow(10,6) )

    else:
        pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
                          (pz - qz*zp0)*1)*pow(10,5) ) >= ((pixelX)*qx*(pz - qz*zp0)*1)*pow(10,6) )
        pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
                          (pz - qz*zp0)*1)*pow(10,5) ) < (((pixelX+1))*qx*(pz - qz*zp0)*1)*pow(10,6) )

        pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
                          (pz - qz*zp0)*1)*pow(10,5) ) >= ((pixelY)*qy*(pz - qz*zp0)*1)*pow(10,6) )
        pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
                          (pz - qz*zp0)*1)*pow(10,5) ) < (((pixelY+1))*qy*(pz - qz*zp0)*1)*pow(10,6) )

    
    
    

def addAvertexPixelConstraint(currZP, vertexNumber, pixelX, pixelY, pd):
    x = vertices[vertexNumber*3+0]
    y = vertices[vertexNumber*3+1]
    z = vertices[vertexNumber*3+2]
    if (isinstance(x, float)):
        # print("float x")
        xf = str(Fraction(x).limit_denominator())
        xl = xf.split('/')
        px = int(xl[0])
        # qx = int(xl[1])
        if(len(xl) == 2):
            qx = int(xl[1])
        else:
            qx = 1

    else:
        px = x
        qx = 1
    if (isinstance(y, float)):
        # print("float y")
        yf = str(Fraction(y).limit_denominator())
        yl = yf.split('/')
        py = int(yl[0])
        # qy = int(yl[1])
        if(len(yl) == 2):
            qy = int(yl[1])
        else:
            qy = 1
    else:
        py = y
        qy = 1
    if (isinstance(z, float)):
        # print("float z")
        zf = str(Fraction(z).limit_denominator())
        zl = zf.split('/')
        pz = int(zl[0])
        # qz = int(zl[1])
        if(len(zl) == 2):
            qz = int(zl[1])
        else:
            qz = 1
    else:
        pz = z
        qz = 1
    # xf = str(Fraction(x).limit_denominator())
    # yf = str(Fraction(y).limit_denominator())
    # zf = str(Fraction(z).limit_denominator())
    # xl = xf.split('/')
    # yl = yf.split('/')
    # zl = zf.split('/')
    # px = int(xl[0])
    # qx = int(xl[1])
    # py = int(yl[0])
    # qy = int(yl[1])
    # pz = int(zl[0])
    # qz = int(zl[1])

    # print("adding inside vertex cons vertexnumber : ",vertexNumber,pixelX,pixelY)
    # print(x,y,z)
    # print(px, qx, py, qy, pz, qz)

    if(z - currZP < 0):
        # print("z - currZP <0")
        pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
                            (pz - qz*zp0)*1)*pow(10,5) ) <= ((pixelX)*qx*(pz - qz*zp0)*1)*pow(10,6) )
        
        
        pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
                          (pz - qz*zp0)*1)*pow(10,5) ) > (((pixelX+1))*qx*(pz - qz*zp0)*1)*pow(10,6) )

        pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
                          (pz - qz*zp0)*1)*pow(10,5) ) <= ((pixelY)*qy*(pz - qz*zp0)*1)*pow(10,6) )
        pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
                          (pz - qz*zp0)*1)*pow(10,5) ) > (((pixelY+1))*qy*(pz - qz*zp0)*1)*pow(10,6) )

    else:
        pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
                          (pz - qz*zp0)*1)*pow(10,5) ) >= ((pixelX)*qx*(pz - qz*zp0)*1)*pow(10,6) )
        pd.add_constraint(((-6839567*qz*(px - qx*xp0))*pow(10,1) + (245*qx *
                          (pz - qz*zp0)*1)*pow(10,5) ) < (((pixelX+1))*qx*(pz - qz*zp0)*1)*pow(10,6) )

        pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
                          (pz - qz*zp0)*1)*pow(10,5) ) >= ((pixelY)*qy*(pz - qz*zp0)*1)*pow(10,6) )
        pd.add_constraint(((6839567*qz*(py - qy*yp0))*pow(10,1) + (245*qy *
                          (pz - qz*zp0)*1)*pow(10,5) ) < (((pixelY+1))*qy*(pz - qz*zp0)*1)*pow(10,6) )


def getCurrentPosOutcodeCons2(outcodeP0, pd):

    mproj_1_1 = int(float((2 * n / (t - b)))*pow(10, 58)//1)

    mproj_0_0 = int(float((2 * n / (r - l)))*pow(10, 58)//1)

    mproj_2_2 = int(float((-(f + n) / (f - n)))*pow(10, 58)//1)
    mproj_3_2 = int(float((-2 * f * n / (f - n)))*pow(10, 58)//1)

    # print(mproj_0_0)
    #print((2 * n / (r - l)))
    # print(mproj_1_1)
    #print((2 * n / (t - b)))

    # print(mproj_2_2)
    #print((-(f + n) / (f - n)))
    # print(mproj_3_2)
    #print((-2 * f * n / (f - n)))

    for i in range(0, numOfVertices):
        print(i)
        x = vertices[i*3+0]
        y = vertices[i*3+1]
        z = vertices[i*3+2]
        print(i, ": ", x, y, z)
        if (isinstance(x, float)):
            xf = str(Fraction(x).limit_denominator())
            xl = xf.split('/')
            print(xl)
            px = int(xl[0])
            if(len(xl) == 2):
                qx = int(xl[1])
            else:
                qx = 1
        else:
            px = x
            qx = 1
        if (isinstance(y, float)):
            yf = str(Fraction(y).limit_denominator())
            yl = yf.split('/')
            py = int(yl[0])
            # qy = int(yl[1])
            if(len(yl) == 2):
                qy = int(yl[1])
            else:
                qy = 1
        else:
            py = y
            qy = 1
        if (isinstance(z, float)):
            zf = str(Fraction(z).limit_denominator())
            zl = zf.split('/')
            pz = int(zl[0])
            # qz = int(zl[1])
            if(len(zl) == 2):
                qz = int(zl[1])
            else:
                qz = 1

        else:
            pz = z
            qz = 1

        if(outcodeP0[i*6+0] == 0):
            #pd.add_constraint( (vertices[i*3+1] -yp0) * mproj_1_1   <=      -(vertices[i*3+2] -zp0)*pow(10,58) )
            pd.add_constraint(qz*(py - qy*yp0) * mproj_1_1 <= -
                              (qy)*(pz - qz*zp0)*pow(10, 58))
        else:
            # pd.add_constraint( (vertices[i*3+1] -yp0) * mproj_1_1  >   -(vertices[i*3+2] -zp0) *pow(10,58) )
            pd.add_constraint(qz*(py - qy*yp0) * mproj_1_1 > -
                              (qy)*(pz - qz*zp0) * pow(10, 58))

        if(outcodeP0[i*6+1] == 0):
            # pd.add_constraint( ((   (vertices[i*3+2] -zp0)*pow(10,58)  ) <=      (vertices[i*3+1] -yp0) * mproj_1_1 ) )
            pd.add_constraint(((qy*(pz - qz*zp0)*pow(10, 58))
                              <= qz*(py - qy*yp0) * mproj_1_1))
        else:
            # pd.add_constraint( ( (  (vertices[i*3+2] -zp0)*pow(10,58) ) >   (vertices[i*3+1] -yp0) * mproj_1_1 )  )
            pd.add_constraint(((qy*(pz - qz*zp0)*pow(10, 58))
                              > qz*(py - qy*yp0) * mproj_1_1))

        if(outcodeP0[i*6+2] == 0):
            #    pd.add_constraint( ( (vertices[i*3+0] -xp0) *mproj_0_0  <=     -(vertices[i*3+2] -zp0)*pow(10,58) ) )
            pd.add_constraint(
                (qz*(px - qx*xp0) * mproj_0_0 <= -(qx)*(pz - qz*zp0)*pow(10, 58)))
        else:
            # pd.add_constraint( ( (vertices[i*3+0] -xp0) *mproj_0_0  >    -(vertices[i*3+2] -zp0)*pow(10,58) )  )
            pd.add_constraint(
                (qz*(px - qx*xp0) * mproj_0_0 > -(qx)*(pz - qz*zp0)*pow(10, 58)))

        if(outcodeP0[i*6+3] == 0):
            # pd.add_constraint( ( (  (vertices[i*3+2] -zp0)*pow(10,58)  )<=      (vertices[i*3+0] -xp0) *mproj_0_0 ) )
            pd.add_constraint(((qx*(pz - qz*zp0)*pow(10, 58))
                              <= qz*(px - qx*xp0) * mproj_0_0))
        else:
            # pd.add_constraint( ( (  (vertices[i*3+2] -zp0)*pow(10,58)  )>   (vertices[i*3+0] -xp0) *mproj_0_0 ) )
            pd.add_constraint(((qx*(pz - qz*zp0)*pow(10, 58))
                              > qz*(px - qx*xp0) * mproj_0_0))

        if(outcodeP0[i*6+4] == 0):
            # pd.add_constraint( (  (vertices[i*3+2] -zp0) * mproj_2_2 + mproj_3_2 <=  -(vertices[i*3+2] -zp0) *pow(10,58)) )
            pd.add_constraint(((pz - qz*zp0) * mproj_2_2 +
                              qz*mproj_3_2 <= -(pz - qz*zp0) * pow(10, 58)))
        else:
            # pd.add_constraint( (  (vertices[i*3+2] -zp0) * mproj_2_2 + mproj_3_2 > -(vertices[i*3+2] -zp0) *pow(10,58) ) )
            pd.add_constraint(((pz - qz*zp0) * mproj_2_2 +
                              qz*mproj_3_2 > -(pz - qz*zp0) * pow(10, 58)))

        if(outcodeP0[i*6+5] == 0):
            # pd.add_constraint( ( (   (vertices[i*3+2] -zp0) *pow(10,58)  ) <=    (vertices[i*3+2] -zp0) * mproj_2_2 + mproj_3_2) )
            pd.add_constraint((((pz - qz*zp0) * pow(10, 58))
                              <= (pz - qz*zp0) * mproj_2_2 + qz*mproj_3_2))
        else:
            # pd.add_constraint( ( (   (vertices[i*3+2] -zp0)*pow(10,58)  ) >   (vertices[i*3+2] -zp0) * mproj_2_2 + mproj_3_2) )
            pd.add_constraint((((pz - qz*zp0)*pow(10, 58)) >
                              (pz - qz*zp0) * mproj_2_2 + qz*mproj_3_2))


# ####################### Old idea #############################

def intersectingEdgeRegion(planeId, edgeId, insideVertex,outsideVertex, xPixel, yPixel, pd):
    
    
    r =0.35821;

    
    x1 = vertices[insideVertex*3 +0];
    y1 = vertices[insideVertex*3 +1];
    z1 = vertices[insideVertex*3 +2];

    x0 = vertices[outsideVertex*3 +0];
    y0 = vertices[outsideVertex*3 +1];
    z0 = vertices[outsideVertex*3 +2];
    
    fraction_constant = 0.01462077368
    fraction_constant_far = 14.62077368
    
    nearCorner = .35820895522388063
    farCorner = 358.20895522388063
    
    pd2 =NNC_Polyhedron(3,'empty')
    
    if planeId == 0 or planeId == 1:
        # print("top plane")
        
        #nearplane edge's top vertex
        BF_top_right_x = (x0+nearCorner)- (xPixel * fraction_constant);
        BF_top_left_x = (x0+nearCorner)- ((xPixel+1) * fraction_constant);
        #nearplane edge's bottom vertex
        BF_bot_right_x = (x1+nearCorner)- (xPixel * fraction_constant);
        BF_bot_left_x = (x1+nearCorner)- ((xPixel+1) *fraction_constant);

        

        BF_top_z = (z0+1);
        BF_bot_z = (z1+1);

        FF_top_right_x = (x0+farCorner)- (xPixel * fraction_constant_far);
        FF_top_left_x = (x0+farCorner)- ((xPixel+1) * fraction_constant_far);
        FF_bot_right_x = (x1+farCorner)- (xPixel * fraction_constant_far);
        FF_bot_left_x = (x1+farCorner)- ((xPixel+1) *fraction_constant_far);



        FF_top_z = (z0+1000);
        FF_bot_z = (z1+1000);
        
        if planeId == 0:
            BF_top_y = (y0-nearCorner);
            BF_bot_y = (y1-nearCorner);
            
            FF_top_y = (y0-farCorner);
            FF_bot_y = (y1-farCorner);
            
            
        else:
            BF_top_y = (y0+nearCorner);
            BF_bot_y = (y1+nearCorner);
            
            FF_top_y = (y0+farCorner);
            FF_bot_y = (y1+farCorner);

        i_BF_top_right_x = int(BF_top_right_x*pow(10,15)//1)
        print("BF_top_right_x = ", BF_top_right_x," : ", i_BF_top_right_x)


        i_BF_top_left_x = int( BF_top_left_x*pow(10,15)//1)
        print("BF_top_left_x = ", BF_top_left_x," : ", i_BF_top_left_x)


        i_BF_bot_right_x = int( BF_bot_right_x*pow(10,15)//1)
        print("BF_bot_right_x = ", BF_bot_right_x," : ", i_BF_bot_right_x)

        i_BF_bot_left_x = int( BF_bot_left_x*pow(10,15)//1)
        print("BF_bot_left_x = ", BF_bot_left_x," : ", i_BF_bot_left_x)

        i_BF_top_y = int(BF_top_y*pow(10,15)//1)
        print("BF_top_y = ", BF_top_y," : ", i_BF_top_y)


        i_BF_bot_y = int( BF_bot_y*pow(10,15)//1)
        print("BF_bot_y = ", BF_bot_y," : ", i_BF_bot_y)

        i_BF_top_z = int( BF_top_z*pow(10,15)//1)
        print("BF_top_z = ", BF_top_z," : ", i_BF_top_z)

        i_BF_bot_z = int( BF_bot_z *pow(10,15)//1)
        print("BF_bot_z = ", BF_bot_z," : ", i_BF_bot_z)

        print("\n\n")

        i_FF_top_right_x = int(FF_top_right_x*pow(10,15)//1)
        print("FF_top_right_x = ", FF_top_right_x," : ", i_FF_top_right_x)

        i_FF_top_left_x = int( FF_top_left_x*pow(10,15)//1)
        print("FF_top_left_x = ", FF_top_left_x," : ", i_FF_top_left_x)

        i_FF_bot_right_x =  int( FF_bot_right_x*pow(10,15)//1)
        print("FF_bot_right_x = ", FF_bot_right_x," : ", i_FF_bot_right_x)

        i_FF_bot_left_x =  int( FF_bot_left_x*pow(10,15)//1)
        print("FF_bot_left_x = ", FF_bot_left_x," : ", i_FF_bot_left_x)

        i_FF_top_y =  int( FF_top_y*pow(10,15)//1)
        print("FF_top_y = ", FF_top_y," : ", i_FF_top_y)


        i_FF_bot_y =  int( FF_bot_y*pow(10,15)//1)
        print("FF_bot_y = ", FF_bot_y," : ", i_FF_bot_y)

        i_FF_top_z =  int( FF_top_z*pow(10,15)//1)
        print("FF_top_z = ", FF_top_z," : ", i_FF_top_z)

        i_FF_bot_z =  int( FF_bot_z*pow(10,15)//1)
        print("FF_bot_z = ", FF_bot_z," : ", i_FF_bot_z)


        print("\n\n")

        
        pd2.add_generator(point( i_BF_top_right_x*xp0+i_BF_top_y*yp0+i_BF_top_z*zp0 ,pow(10,15)))
        pd2.add_generator(point( i_BF_top_left_x*xp0+ i_BF_top_y*yp0+ i_BF_top_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_BF_bot_right_x*xp0+ i_BF_bot_y*yp0+ i_BF_bot_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_BF_bot_left_x*xp0+ i_BF_bot_y*yp0+ i_BF_bot_z*zp0,pow(10,15)))

        pd2.add_generator(point(i_FF_top_right_x*xp0+ i_FF_top_y*yp0+  i_FF_top_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_FF_top_left_x*xp0+ i_FF_top_y*yp0+ i_FF_top_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_FF_bot_right_x*xp0+ i_FF_bot_y*yp0+ i_FF_bot_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_FF_bot_left_x*xp0+ i_FF_bot_y*yp0+ i_FF_bot_z*zp0,pow(10,15)))

        
   
        
    if planeId == 2 or planeId ==3:
        # print("right plane")
        plane0_v0 = [0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, -358.20895522388063, -1000]
        
        if planeId == 2:
            BF_rVertex_x = x0 - nearCorner
            BF_lVertex_x = x0 - nearCorner
            
            FF_rVertex_x = x0 - farCorner
            FF_lVertex_x = x0 - farCorner
        
        else:
            
            BF_rVertex_x = x0 + nearCorner
            BF_lVertex_x = x0 + nearCorner
            
            FF_rVertex_x = x0 + farCorner
            FF_lVertex_x = x0 + farCorner
            
        BF_rVertex_top_y = (y0-nearCorner)+ (yPixel * fraction_constant)
        BF_rVertex_bot_y = (y0-nearCorner)+ ( (yPixel+1) * fraction_constant)
        BF_lVertex_top_y = (y1-nearCorner)+ (yPixel * fraction_constant)
        BF_lVertex_bot_y = (y1-nearCorner)+ ( (yPixel+1) * fraction_constant)
        
        

        BF_rVertex_z = (z0+1);
        BF_lVertex_z = (z1+1);
        
        FF_rVertex_top_y = (y0-farCorner)+ (yPixel * fraction_constant_far)
        FF_rVertex_bot_y = (y0-farCorner)+ ( (yPixel+1) * fraction_constant_far)
        FF_lVertex_top_y = (y1-farCorner)+ (yPixel * fraction_constant_far)
        FF_lVertex_bot_y = (y1-farCorner)+ ( (yPixel+1) * fraction_constant_far)
      
      

        

        FF_rVertex_z = (z0+1000);
        FF_lVertex_z = (z1+1000);
        
        
        i_BF_rVertex_x = int(BF_rVertex_x*pow(10,15)//1)
        i_BF_lVertex_x = int(BF_lVertex_x*pow(10,15)//1)
        i_FF_rVertex_x = int( FF_rVertex_x*pow(10,15)//1)
        i_FF_lVertex_x = int( FF_lVertex_x*pow(10,15)//1)
        
        
        i_BF_rVertex_top_y = int( BF_rVertex_top_y*pow(10,15)//1)
        i_BF_rVertex_bot_y = int( BF_rVertex_bot_y*pow(10,15)//1)
        i_BF_lVertex_top_y = int( BF_lVertex_top_y*pow(10,15)//1)
        i_BF_lVertex_bot_y = int(BF_lVertex_bot_y*pow(10,15)//1)
        
        
         
        
        i_FF_rVertex_top_y = int( FF_rVertex_top_y*pow(10,15)//1)
        i_FF_rVertex_bot_y = int( FF_rVertex_bot_y*pow(10,15)//1)
        i_FF_lVertex_top_y = int( FF_lVertex_top_y*pow(10,15)//1)
        i_FF_lVertex_bot_y = int( FF_lVertex_bot_y*pow(10,15)//1)
        
      
        i_BF_rVertex_z = int( BF_rVertex_z*pow(10,15)//1)
        i_BF_lVertex_z = int( BF_lVertex_z*pow(10,15)//1)
        
        i_FF_rVertex_z = int( FF_rVertex_z*pow(10,15)//1)
        i_FF_lVertex_z = int( FF_lVertex_z*pow(10,15)//1)
        

        
        
        print("\n\n")

        
        pd2.add_generator(point( i_BF_rVertex_x*xp0+i_BF_rVertex_top_y*yp0+i_BF_rVertex_z*zp0 ,pow(10,15)))
        pd2.add_generator(point( i_BF_rVertex_x*xp0+ i_BF_rVertex_bot_y*yp0+ i_BF_rVertex_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_BF_lVertex_x*xp0+ i_BF_lVertex_top_y*yp0+ i_BF_lVertex_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_BF_lVertex_x*xp0+ i_BF_lVertex_bot_y*yp0+ i_BF_lVertex_z*zp0,pow(10,15)))

        # pd2.add_generator(point(i_FF_top_right_x*xp0+ i_FF_top_y*yp0+  i_FF_top_z*zp0,pow(10,15)))
        # pd2.add_generator(point( i_FF_top_left_x*xp0+ i_FF_top_y*yp0+ i_FF_top_z*zp0,pow(10,15)))
        # pd2.add_generator(point( i_FF_bot_right_x*xp0+ i_FF_bot_y*yp0+ i_FF_bot_z*zp0,pow(10,15)))
        # pd2.add_generator(point( i_FF_bot_left_x*xp0+ i_FF_bot_y*yp0+ i_FF_bot_z*zp0,pow(10,15)))

        pd2.add_generator(point( i_FF_rVertex_x*xp0+i_FF_rVertex_top_y*yp0+i_FF_rVertex_z*zp0 ,pow(10,15)))
        pd2.add_generator(point( i_FF_rVertex_x*xp0+ i_FF_rVertex_bot_y*yp0+ i_FF_rVertex_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_FF_lVertex_x*xp0+ i_FF_lVertex_top_y*yp0+ i_FF_lVertex_z*zp0,pow(10,15)))
        pd2.add_generator(point( i_FF_lVertex_x*xp0+ i_FF_lVertex_bot_y*yp0+ i_FF_lVertex_z*zp0,pow(10,15)))

        
        
        
        
       
    if planeId == 5:
        plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [-0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v3 = [0.35820895522388063, 0.35820895522388063, -1]
        
        
        print("TODO: near plane pyparma 34")
        print("exiting")
        exit()
       



    

    print("\n\n")

    
    print(pd2.generators())
    print(pd2.minimized_constraints())          
    
           

    # pd.intersection_assign(pd2);
    return pd2





###################### old idea end here ######################
def findPosUpdated(x0, y0, z0, nearFar, xpixel, ypixel, planeId,intersectionPoint=0):
    
    
    
    s = Solver()
    set_option(rational_to_decimal=True)
    set_option(precision=105)
    s.set("timeout", 10000)

    xp, yp, zp = Reals('xp yp zp')
    a, b = Reals('a b')

    xk, yk, zk = Reals('xk yk zk')
    u, v, w, g = Reals('u v w g')    
    c,d = Reals('c d')
    
    tpc, bpc, rpc, lpc = Reals('tpc bpc rpc lpc') 
    
    s.add(u+v == 1)
    s.add(And(u >= 0, v >= 0))
    # s.add(And(b >= 0, b <= 0.01))

    plane0_v0 = [0, 0.0, -1]
    plane0_v1 = [0.0, 0.0, -1]
    plane0_v2 = [0, 0, -1000]
    plane0_v3 = [0, 0, -1000]

    cons1 = ""
    cons2 = ""

    print("x0,y0,z0,nearFar,xpixel,ypixel,planeId :",
          x0, y0, z0, nearFar, xpixel, ypixel, planeId)

    # if xpixel == 49:
    #     xpixel =48.99
    # if ypixel == 49:
    #     ypixel =48.99
    # s4 = Solver()
    # h,f = Reals('h f')
    # s4.add(h+f == 1)
    # s4.add(And(h>=0,f>=0))  
    
    
    # s4.add(And(c>=0, c<49))
    # s4.add(And(d>=48.9999, d<=49))
    
    # s4.add(xv0-xp0 == (h*x2+f*x3))
    # s4.add(yv0-yp0 == (h*y2+f*y3))
    # s4.add(zv0-zp0 == (h*z2+f*z3))
    
    # print(s4.check())
    # m3 = s4.model()
    # print(m3)
    
    # cons1 = ( (((-68.39567*(xv0-xp0 ) )/ (zv0-zp0) )+ 24.5 ) == 26 )
    # cons2 =  ( (((68.39567*(yv0-yp0 ) )/ (zv0-zp0) )+ 24.5 ) == d )
    # s4.add((And(cons1,cons2)))
    
    # print(s4.check())
    # m = s4.model()
    # print(m)
    
    


    if planeId == 0:
        # print("top plane")
        plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, 0.35820895522388063, -1]
        plane0_v2 = [-358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, 358.20895522388063, -1000]
        # plane0_v0 = [-0.3583, 0.35820895522388063, -1]
        # plane0_v1 = [0.3583, 0.35820895522388063, -1]
        # plane0_v2 = [-358.208956, 358.20895522388063, -1000]
        # plane0_v3 = [358.208956, 358.20895522388063, -1000]
        
        
        # s3 = Solver()
        
        # s3.add(u+v == 1)
        # s3.add(And(u >= 0, v >= 0))
    
    
        # s3.add(xk == (u*x0+v*x1+w*x2+g*x3))
        # s3.add(yk == (u*y0+v*y1+w*y2+g*y3))
        # s3.add(zk == (u*z0+v*z1+w*z2+g*z3))
        # s3.check() 
        
        # ypixel = 0
        
    if planeId == 1:
        print("bottom plane")
        plane0_v0 = [-0.35820895522388063, -0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [-358.20895522388063, -358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, -358.20895522388063, -1000]
        
        # ypixel =49
        
    if planeId == 2:
        # print("right plane")
        plane0_v0 = [0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, -358.20895522388063, -1000]
        
        # xpixel =49
        
    if planeId == 3:
        print("left plane ")
        plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [-0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [-358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [-358.20895522388063, -358.20895522388063, -1000]
        # cons1 = "0-b == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
        
        # xpixel = 0
        
    if planeId == 5:
        plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [-0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v3 = [0.35820895522388063, 0.35820895522388063, -1]
        cons1 = "xpixel == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
        cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zpos))+24.5"

    px0 = plane0_v0[0]
    py0 = plane0_v0[1]
    pz0 = plane0_v0[2]

    px1 = plane0_v1[0]
    py1 = plane0_v1[1]
    pz1 = plane0_v1[2]

    px2 = plane0_v2[0]
    py2 = plane0_v2[1]
    pz2 = plane0_v2[2]

    px3 = plane0_v3[0]
    py3 = plane0_v3[1]
    pz3 = plane0_v3[2]
    
    
    if(nearFar == 1):
        s.add(x0-xp == (u*px0+(1-u)*px1))
        s.add(y0-yp == (u*py0+(1-u)*py1))
        s.add(z0-zp == (u*pz0+(1-u)*pz1))      
        
    elif(nearFar ==1000):
        s.add(x0-xp == (u*px2+v*px3))
        s.add(y0-yp == (u*py2+v*py3))
        s.add(z0-zp == (u*pz2+v*pz3))
        
    else:
        print("Unknown zpos, nearFar")
        exit()
        
    print(s.check())
    
    if planeId == 0:
        print("top")
        # s.add(And(a>=0, a<=0.0000000000000000001))
        # if ypixel == 0:
        #     s.add(And(b>=0, b<=0.0000000000000001))
        # else:
        #     s.add(b == ypixel)  
        # if xpixel == 0:
        #     s.add(And(a>=0, a<=0.0000000000001))
        # else:
        #     s.add(a == xpixel)        
        # s.add(And(b>=0, b<=0.01))
        
        s3 = Solver()        
        s3.add(u+v == 1)
        s3.add(And(u >= 0, v >= 0))   
        if(nearFar == 1):
            s3.add(x0-xp == (u*px0+(1-u)*px1))
            s3.add(y0-yp == (u*py0+(1-u)*py1))
            s3.add(z0-zp == (u*pz0+(1-u)*pz1))      
        
        elif(nearFar ==1000):
            s3.add(x0-xp == (u*px2+v*px3))
            s3.add(y0-yp == (u*py2+v*py3))
            s3.add(z0-zp == (u*pz2+v*pz3))
        
        newYPixel = Real('newYPixel')
        cons20 = "newYPixel == ((68.39567*(y0-yp))/(z0-zp))+24.5" 
        s3.add(eval(cons20))        
        print(s3.check())
        m3= s3.model()
        print(m3)
        
        
        if xpixel == 49:
            xpixel = 48.99
        
        
        cons1 = "xpixel == ((-68.39567*(x0-xp))/(z0-zp))+24.5"
        cons2 = "m3[newYPixel] == ((68.39567*(y0-yp))/(z0-zp))+24.5"       
        s.add((And(eval(cons1),eval(cons2))) )    
        print(s.check())  
        
        
        
    elif planeId == 1:
        print("bot")
        # if xpixel == 0:
        #     s.add(And(a>=0, a<=0.0000001))
        # else:
        #     s.add(a == xpixel)        
        # s.add(And(b>=48.99, b<=49))
        
        s3 = Solver()        
        s3.add(u+v == 1)
        s3.add(And(u >= 0, v >= 0))   
        if(nearFar == 1):
            s3.add(x0-xp == (u*px0+v*px1))
            s3.add(y0-yp == (u*py0+v*py1))
            s3.add(z0-zp == (u*pz0+v*pz1))      
        
        elif(nearFar ==1000):
            s3.add(x0-xp == (u*px2+v*px3))
            s3.add(y0-yp == (u*py2+v*py3))
            s3.add(z0-zp == (u*pz2+v*pz3))
        
        newYPixel = Real('newYPixel')
        cons20 = "newYPixel == ((68.39567*(y0-yp))/(z0-zp))+24.5" 
        s3.add(eval(cons20))        
        print(s3.check())
        m3= s3.model()
        print(m3)
        
        
        if xpixel == 49:
            xpixel = 48.99
        
        
        cons1 = "xpixel == ((-68.3956*(x0-xp))/(z0-zp))+24.5"
        cons2 = "m3[newYPixel] == ((68.39567*(y0-yp))/(z0-zp))+24.5"       
        s.add((And(eval(cons1),eval(cons2))) )
        
    elif planeId == 2:
        print("right")
        # s.add(And(a>=48.9999, a<=49))
        # # if ypixel == 0:
        #     s.add(And(b>=0, b<=0.01))
        # else:
        #     s.add(b == ypixel)  
        # s.add(And(b>=48.99, b<=49))
        
        s3 = Solver()        
        s3.add(u+v == 1)
        s3.add(And(u >= 0, v >= 0))   
        if(nearFar == 1):
            s3.add(x0-xp == (u*px0+v*px1))
            s3.add(y0-yp == (u*py0+v*py1))
            s3.add(z0-zp == (u*pz0+v*pz1))      
        
        elif(nearFar ==1000):
            s3.add(x0-xp == (u*px2+v*px3))
            s3.add(y0-yp == (u*py2+v*py3))
            s3.add(z0-zp == (u*pz2+v*pz3))
        
        newXPixel = Real('newXPixel')
        cons20 = "newXPixel == ((-68.39567*(x0-xp))/(z0-zp))+24.5" 
        s3.add(eval(cons20))        
        print(s3.check())
        m3= s3.model()
        print(m3)
        
        if ypixel == 49:
            ypixel = 48.99
        cons1 = "m3[newXPixel] == ((-68.39567*(x0-xp))/(z0-zp))+24.5"
        cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zp))+24.5"       
        s.add((And(eval(cons1),eval(cons2))) )        
        
    elif planeId == 3:
        print("left")
        # s.add(And(a>=0, a<=0.0001))
        # if ypixel == 0:
        #     s.add(And(b>=0, b<=0.0001))
        # else:
        #     s.add(b == ypixel)     
        # s.add(And(b>=0, b<=0.01))  
        
        s3 = Solver()        
        s3.add(u+v == 1)
        s3.add(And(u >= 0, v >= 0))   
        if(nearFar == 1):
            s3.add(x0-xp == (u*px0+v*px1))
            s3.add(y0-yp == (u*py0+v*py1))
            s3.add(z0-zp == (u*pz0+v*pz1))      
        
        elif(nearFar ==1000):
            s3.add(x0-xp == (u*px2+v*px3))
            s3.add(y0-yp == (u*py2+v*py3))
            s3.add(z0-zp == (u*pz2+v*pz3))
        
        newXPixel = Real('newXPixel')
        cons20 = "newXPixel == ((-68.39567*(x0-xp))/(z0-zp))+24.5" 
        s3.add(eval(cons20))        
        print(s3.check())
        m3= s3.model()
        print(m3)
        
        if ypixel == 49:
            ypixel = 48.99
        
        cons1 = "m3[newXPixel] == ((-68.39567*(x0-xp))/(z0-zp))+24.5"
        cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zp))+24.5"       
        s.add((And(eval(cons1),eval(cons2))) )


    print("Intersecting or not")
    print(s.check())
    m= s.model()
    # print(m)

    # print( ((-68.39567*(x0-m[xp]))/(z0-zpos))+24.5 ) 
    # print( ((68.39567*(y0-m[yp]))/(z0-zpos))+24.5 ) 
    # # cons1 = "xpixel == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
    # # cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zpos))+24.5"

    # print(eval(cons1))
    # print(eval(cons2))
    
    # s.add(eval(cons1))
    # print(s.check())
    # s.add(eval(cons2))
    # print(s.check())
    # s.add(And(eval(cons1),eval(cons2)))

    # print()
    result = s.check()
    print(result)
    if result == sat:
        m = s.model()
        print(m)
        # print("\n")

        posx = str(m[xp]).replace("?","")
        posy = str(m[yp]).replace("?","")
        posz = str(m[zp]).replace("?","")

        # posx = m[xp]
        # posy = m[yp]
        # posz = m[zp]
        # posz = zpos
        
        print("Intersecting point on the line = ", float(str(m[u]).replace("?",""))*px0+(1-float(str(m[u]).replace("?",""))*px1))

        print(m[xp],m[yp],m[zp])
        print("\n\n")
        print(posx,posy,posz)
        
        # sleep(3)

        # pointPos.append([posx,posy,posz])
        # print(pointPos)
        # print("\n\n")
        
        # exit()

        del(s)
        return [posx, posy, posz]
    if result == unknown:
        print("timeout occured trying again")
        del(s)
        sleep(10)
        findPos(x0, y0, z0, zpos, xpixel, ypixel, planeId)
    else:
        print(result)
        print("x0,y0,z0,zpos,xpixel,ypixel,planeId :",
              x0, y0, z0, xpixel, ypixel, planeId)
        print("no matching solution found, ")
        del(s)
        global unsatFlag
        unsatFlag = 1
        exit()
        sleep(3)
        return 0







def findPos(x0, y0, z0, zpos, xpixel, ypixel, planeId,intersectionPoint=0):

    s = Solver()
    set_option(rational_to_decimal=True)
    set_option(precision=105)
    s.set("timeout", 10000)

    xp, yp, zp = Reals('xp yp zp')
    b = Real('b')

    xk, yk, zk = Reals('xk yk zk')
    u, v, w, g = Reals('u v w g')

    s.add(And(b >= 0, b <= .00001))
    s.add(u+v+w+g == 1)
    s.add(And(u >= 0, v >= 0, w >= 0, g >= 0))

    plane0_v0 = [0, 0.0, -1]
    plane0_v1 = [0.0, 0.0, -1]
    plane0_v2 = [0, 0, -1000]
    plane0_v3 = [0, 0, -1000]

    cons1 = ""
    cons2 = ""

    print("x0,y0,z0,zpos,xpixel,ypixel,planeId :",
          x0, y0, z0, zpos, xpixel, ypixel, planeId)

    if planeId == 0:
        # print("top plane")
        plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, 0.35820895522388063, -1]
        plane0_v2 = [-358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, 358.20895522388063, -1000]
        cons1 = "xpixel == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
        if xpixel < 0:
            xpixel = 0
        elif xpixel > 48:
            xpixel = 48.9
        cons1 = "xpixel == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
        if ypixel != 0:
            # print("ypixel not zero")
            cons2 = "0-b == ((68.39567*(y0-yp))/(z0-zpos))+24.5"
        else:
            cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zpos))+24.5"

    if planeId == 1:
        print("bottom plane")
        plane0_v0 = [-0.35820895522388063, -0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [-358.20895522388063, -358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, -358.20895522388063, -1000]
        if xpixel < 0:
            xpixel = 0
        elif xpixel > 48:
            xpixel = 48.9
        cons1 = "xpixel == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
        # cons2 = "48-b == ((68.39567*(y0-yp))/(z0-zpos))+24.5"
        if ypixel != 48:
            # print("ypixel not 48")
            cons2 = "48.00000000000003-b == ((68.39567*(y0-yp))/(z0-zpos))+24.5"
        else:
            cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zpos))+24.5"

    if planeId == 2:
        # print("right plane")
        plane0_v0 = [0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, -358.20895522388063, -1000]
        # cons1 = "48-b == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
        if xpixel != 48:
            cons1 = "48.00000000000003-b == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
        else:
            cons1 = "xpixel == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
        if ypixel > 48:
            #print("ypixel >48")
            ypixel = 48.9
        elif ypixel < 0:
            ypixel = 0
        cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zpos))+24.5"
    if planeId == 3:
        print("left plane ")
        plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [-0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [-358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [-358.20895522388063, -358.20895522388063, -1000]
        # cons1 = "0-b == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
        if xpixel != 0:
            # print("xpixel not zero")
            cons1 = "0-b == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
        else:
            cons1 = "xpixel == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
        if ypixel > 48:
            #print("ypixel >48")
            ypixel = 48.9
        elif ypixel < 0:
            ypixel = 0
        cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zpos))+24.5"
    if planeId == 5:
        plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [-0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v3 = [0.35820895522388063, 0.35820895522388063, -1]
        cons1 = "xpixel == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
        cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zpos))+24.5"

    px0 = plane0_v0[0]
    py0 = plane0_v0[1]
    pz0 = plane0_v0[2]

    px1 = plane0_v1[0]
    py1 = plane0_v1[1]
    pz1 = plane0_v1[2]

    px2 = plane0_v2[0]
    py2 = plane0_v2[1]
    pz2 = plane0_v2[2]

    px3 = plane0_v3[0]
    py3 = plane0_v3[1]
    pz3 = plane0_v3[2]

    if(intersectionPoint == 1):
        print("Intersection point, already in the camera coordinate system")
        s.add(x0 == (u*px0+v*px1+w*px2+g*px3))
        s.add(y0 == (u*py0+v*py1+w*py2+g*py3))
        s.add(z0 == (u*pz0+v*pz1+w*pz2+g*pz3))
        
        print(s.check())
        cons1 = "xpixel == ((-68.39567*(x0))/(z0))+24.5"
        cons2 = "ypixel == ((68.39567*(y0))/(z0))+24.5"
        
        s.add(eval(cons1))
        s.add(eval(cons2))
        print("intersecting point produces same pixel or not : ",s.check() )
        exit()

    else:
        s.add(x0-xp == (u*px0+v*px1+w*px2+g*px3))
        s.add(y0-yp == (u*py0+v*py1+w*py2+g*py3))
        s.add(z0-zpos == (u*pz0+v*pz1+w*pz2+g*pz3))

    print("Intersecting or not")
    print(s.check())
    m= s.model()
    print(m)

    print( ((-68.39567*(x0-m[xp]))/(z0-zpos))+24.5 ) 
    print( ((68.39567*(y0-m[yp]))/(z0-zpos))+24.5 ) 
    # cons1 = "xpixel == ((-68.39567*(x0-xp))/(z0-zpos))+24.5"
    # cons2 = "ypixel == ((68.39567*(y0-yp))/(z0-zpos))+24.5"

    print(eval(cons1))
    print(eval(cons2))
    
    s.add(eval(cons1))
    print(s.check())
    s.add(eval(cons2))
    print(s.check())
    # s.add(And(eval(cons1),eval(cons2)))

    # print()
    result = s.check()
    if result == sat:
        m = s.model()
        # print(m)
        # print("\n")

        # posx = str(m[xp]).replace("?","")
        # posy = str(m[yp]).replace("?","")
        # posz = str(m[zp]).replace("?","")

        posx = m[xp]
        posy = m[yp]
        # posz = m[zp]
        posz = zpos

        # print(m[xp],m[yp],m[zp])
        # print("\n\n")
        # print(posx,posy,posz)

        # pointPos.append([posx,posy,posz])
        # print(pointPos)
        # print("\n\n")

        del(s)
        return [posx, posy, posz]
    if result == unknown:
        print("timeout occured trying again")
        del(s)
        sleep(10)
        findPos(x0, y0, z0, zpos, xpixel, ypixel, planeId)
    else:
        print(result)
        print("x0,y0,z0,zpos,xpixel,ypixel,planeId :",
              x0, y0, z0, zpos, xpixel, ypixel, planeId)
        print("no matching solution found, ")
        del(s)
        global unsatFlag
        unsatFlag = 1
        exit()
        sleep(3)
        return 0


# -6699932362587579


def computeIntersectingRegionUpdated(planeId, edgeId, insideVertex, outsideVertex, xpixel, ypixel, pd, mxp, myp, mzp, ix, iy, iz, mp, mq,
                              posXp, posYp, posZp):

    x1 = vertices[outsideVertex*3 + 0]
    y1 = vertices[outsideVertex*3 + 1]
    z1 = vertices[outsideVertex*3 + 2]

    x0 = vertices[insideVertex*3 + 0]
    y0 = vertices[insideVertex*3 + 1]
    z0 = vertices[insideVertex*3 + 2]
    
    
    pointsList = []
    # tempList = [posXp,posYp,posZp]
    # pointsList.append(tempList)
    
    # tempList =findPosUpdated(ix,iy,iz,zpos,xpixel,ypixel,planeId, 1)
    # pointsList.append(tempList)
    # tempList =findPosUpdated(ix,iy,iz,zpos,xpixel+1,ypixel,planeId)
    # pointsList.append(tempList)
    near = 1
    far =1000
    
    
    
    
    ########################################
    
    # set_option(rational_to_decimal=False)
    # mxp1 = mxp
    # myp1 = myp
    # mzp1 = mzp
    # mxpStr = [0, 1]
    # mypStr = [0, 1]
    # mzpStr = [0, 1]
    # # Find lcm of denomenators of the pos values
    # if "/" in str(mxp1):
    #     mxpStr = str(mxp1).split("/")
    # else:
    #     mxpStr[0] = str(mxp1)
    #     mxpStr[1] = 1

    # if "/" in str(myp1):
    #     mypStr = str(myp1).split("/")
    # else:
    #     mypStr[0] = str(myp1)
    #     mypStr[1] = 1

    # if "/" in str(mzp1):
    #     mzpStr = str(mzp1).split("/")
    # else:
    #     mzpStr[0] = str(mzp1)
    #     mzpStr[1] = 1

    # numeratorList = [int(str(mxpStr[0])), int(
    #     str(mypStr[0])), int(str(mzpStr[0]))]
    # denomList = [int(int(str(mxpStr[1]))//1),
    #              int(int(str(mypStr[1]))//1), int(int(str(mzpStr[1]))//1)]

    # currLcm = lcm(denomList[0], lcm(denomList[1], denomList[2]))
    # c0 = int(currLcm//(denomList[0]))
    # c1 = int(currLcm//(denomList[1]))
    # c2 = int(currLcm//(denomList[2]))
    # # # print("\nc0,c1,c2 ", c0,c1,c2)

    # pd100 = NNC_Polyhedron(3, 'empty')
    # xp0 = Variable(0)
    # yp0 = Variable(1)
    # zp0 = Variable(2)

    # pd100.add_generator(point( mxp1*xp0+myp1* yp0+mzp1*zp0 ,pow(10,80)))

    # pd100.add_generator(point(
    #     numeratorList[0]*c0 * xp0+numeratorList[1] * c1 * yp0+numeratorList[2] * c2*zp0, int(currLcm)))

    # print("\n pd100.minimized_constraints()")
    # print( pd100.minimized_constraints())

    # print("\n\n (pdC.minimized_constraints()) before mpos add" )
    # print(pdC.minimized_constraints())

    ################################################
    # pdC.poly_hull_assign(pd100)
    ##################################################
    
    # pointsList.append([numeratorList[0]*c0, numeratorList[1] * c1 , numeratorList[2] * c2   ])
    
    pd2 = NNC_Polyhedron(3, 'empty')
   
    # pd2.add_generator(point(
    #     numeratorList[0]*c0 * xp0+numeratorList[1] * c1 * yp0+numeratorList[2] * c2*zp0, int(currLcm)))

    
    
    
    
    
    
    ##########################################
    
    
    # tempList = findPosUpdated(x0, y0, z0, near, xpixel, ypixel, planeId)
    # pointsList.append(tempList)
    # tempList = findPosUpdated(x0, y0, z0, near, xpixel+1, ypixel, planeId)
    # pointsList.append(tempList)
    # tempList = findPosUpdated(x0, y0, z0, far, xpixel, ypixel, planeId)
    # pointsList.append(tempList)
    # tempList = findPosUpdated(x0, y0, z0, far, xpixel+1, ypixel, planeId)
    # pointsList.append(tempList)

    # # tempList =findPosUpdated(ix,iy,iz,zpos,xpixel,ypixel,planeId)
    # # pointsList.append(tempList)
    # # tempList =findPosUpdated(ix,iy,iz,zpos,xpixel+1,ypixel,planeId)
    # # pointsList.append(tempList)
    # tempList = findPosUpdated(x1, y1, z1, near, xpixel, ypixel, planeId)
    # pointsList.append(tempList)
    # tempList = findPosUpdated(x1, y1, z1, near, xpixel+1, ypixel, planeId)
    # pointsList.append(tempList)
    # tempList = findPosUpdated(x1, y1, z1, far, xpixel, ypixel, planeId)
    # pointsList.append(tempList)
    # tempList = findPosUpdated(x1, y1, z1, far, xpixel+1, ypixel, planeId)
    # pointsList.append(tempList)
    
    if planeId == 0 or planeId == 1:
        # top or bottom plane
        print("top/bottom plane")

        # tempList =findPosUpdated(ix,iy,iz,zpos,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPosUpdated(ix,iy,iz,zpos,xpixel+1,ypixel,planeId)
        # pointsList.append(tempList)
        tempList = findPosUpdated(x0, y0, z0, near, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPosUpdated(x0, y0, z0, near,xpixel+1, ypixel, planeId)
        pointsList.append(tempList)
        

        # tempList =findPosUpdated(ix,iy,iz,posZp,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPosUpdated(ix,iy,iz,posZp,xpixel+1,ypixel,planeId)
        # pointsList.append(tempList)
        tempList = findPosUpdated(x1, y1, z1, near, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPosUpdated(x1, y1, z1, near,
                           xpixel+1, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPosUpdated(x1, y1, z1, far, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPosUpdated(x1, y1, z1, far, xpixel+1, ypixel, planeId)
        pointsList.append(tempList)
        
        tempList = findPosUpdated(x0, y0, z0, far, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPosUpdated(x0, y0, z0, far, xpixel+1, ypixel, planeId)
        pointsList.append(tempList)

    if planeId == 2 or planeId == 3:
        print("left/right plane")
        
        # tempList =findPosUpdated(ix,iy,iz,zpos,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPosUpdated(ix,iy,iz,zpos,xpixel,ypixel+1,planeId)
        # pointsList.append(tempList)
        tempList = findPosUpdated(x0, y0, z0, near, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPosUpdated(x0, y0, z0, near, xpixel, ypixel+1, planeId)
        pointsList.append(tempList)
        

        # tempList =findPosUpdated(ix,iy,iz,zpos,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPosUpdated(ix,iy,iz,zpos,xpixel,ypixel+1,planeId)
        # pointsList.append(tempList)
        tempList = findPosUpdated(x1, y1, z1, near, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPosUpdated(x1, y1, z1, near,
                           xpixel, ypixel+1, planeId)
        pointsList.append(tempList)
        tempList = findPosUpdated(x1, y1, z1, far, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPosUpdated(x1, y1, z1, far, xpixel, ypixel+1, planeId)
        pointsList.append(tempList)
        
        tempList = findPosUpdated(x0, y0, z0, far, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPosUpdated(x0, y0, z0, far, xpixel, ypixel+1, planeId)
        pointsList.append(tempList)

        
    if planeId == 5 or planeId == 6:

        print("near far plane pyparma pos inva region 19 ")
        # sleep(10)
        
 # if (float(str(ta2).replace("?","")) > 0.9999):
        #     ta1 = int(ta1)+1
        # else:
        #     ta1 = int(ta1)
        
        # if (float(str(tb2).replace("?","")) > 0.9999):
        #     tb1 = int(tb1)+1
        # else:
        #     tb1 = int(tb1)
        # tempList =findPosUpdated(ix,iy,iz,zpos,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPosUpdated(ix,iy,iz,zpos,xpixel,ypixel+1,planeId)
        # pointsList.append(tempList)
        tempList = findPosUpdated(x0, y0, z0, near, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPosUpdated(x0, y0, z0, near,
                           xpixel, ypixel+1, planeId)
        pointsList.append(tempList)
        

        # tempList =findPosUpdated(ix,iy,iz,zpos,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPosUpdated(ix,iy,iz,zpos,xpixel,ypixel+1,planeId)
        # pointsList.append(tempList)
        tempList = findPosUpdated(x1, y1, z1, near, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPosUpdated(x1, y1, z1, near,
                           xpixel, ypixel+1, planeId)
        pointsList.append(tempList)
        tempList = findPosUpdated(x1, y1, z1, far, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPosUpdated(x1, y1, z1, far, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        
        tempList = findPosUpdated(x0, y0, z0, far, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPosUpdated(x0, y0, z0, far, xpixel, ypixel+1, planeId)
        pointsList.append(tempList)

    
    print("8 points computed exiting")
    

  
    set_option(rational_to_decimal=True)
    for i in [0,1,3,2,6,7,5,4]:
        global unsatFlag
        if(unsatFlag == 1):
            continue
        currPos = pointsList[i]
        # print(currPos)

        # print("currPos")
        mxp1 = currPos[0]
        myp1 = currPos[1]
        mzp1 = currPos[2]

        mxp1 = str(mxp1).replace("?", "")
        myp1 = str(myp1).replace("?", "")
        mzp1 = str(mzp1).replace("?", "")

        ################

        mxp1 = int(float(mxp1)*pow(10, 158))
        myp1 = int(float(myp1)*pow(10, 158))
        mzp1 = int(float(mzp1)*pow(10, 158))

        # print(mxp1,myp1,mzp1)

        pd2.add_generator(point(mxp1*xp0+myp1 * yp0+mzp1*zp0, pow(10, 158)))

    #print(x0, y0, z0,x1,y1,z1)
    # print("\n\nIntersection assign pd, whole invariant region.")
    # print("before intersection")
    # print(pd2.minimized_constraints())
    # pd=pd2
    # pd.intersection_assign(pd2)
    print(pd2.minimized_constraints())
    return pd2


def computeIntersectingRegion2(planeId, edgeId, insideVertex, outsideVertex, xpixel, ypixel, pd, mxp, myp, mzp, ix, iy, iz, mp, mq,
                               posXp, posYp, posZp):

    x0 = vertices[outsideVertex*3 + 0]
    y0 = vertices[outsideVertex*3 + 1]
    z0 = vertices[outsideVertex*3 + 2]

    x1 = vertices[insideVertex*3 + 0]
    y1 = vertices[insideVertex*3 + 1]
    z1 = vertices[insideVertex*3 + 2]

    pdIntersection = NNC_Polyhedron(3)
    xp0 = Variable(0)
    yp0 = Variable(1)
    zp0 = Variable(2)

    xi = Variable(0)
    yi = Variable(1)
    zi = Variable(2)

    p = Variable(0)

    pdIntersection.add_constraint(p >= 0)
    pdIntersection.add_constraint(xp0 >= 0)
    # pdIntersection.add_constraint( xi == p*x0 + (1-p)*x1 )
    # pdIntersection.add_constraint( yi == p*y0 + (1-p)*y1 )
    # pdIntersection.add_constraint( zi == p*z0 + (1-p)*z1 )

    # pdIntersection.add_constraint( ( (-67*(xi - xp0)) + (24*(zi - zp0)*1 ) ) >= ((xpixel)*(zi - zp0)*1)  )
    # pdIntersection.add_constraint( ( (-67*(xi - xp0)) + (24*(zi - zp0)*1 ) ) < (( (xpixel+1))*(zi - zp0)*1)  )

    # pdIntersection.add_constraint( ( (67*qz*(py - qy*yp0) ) + (24*qy*(pz - qz*zp0)*1 ) ) <= ((pixelY)*qy*(pz - qz*zp0)*1)  )
    # pdIntersection.add_constraint( ( (67*qz*(py - qy*yp0) ) + (24*qy*(pz - qz*zp0)*1 ) ) > (( (pixelY+1))*qy*(pz - qz*zp0)*1)  )

    print(pdIntersection.constraints())
    print(pdIntersection.minimized_constraints())
    sleep(300)
    exit(0)
    return pdIntersection

dummyGroupPolyhedra=NNC_Polyhedron(3)
dummyPolyhedraCons = dummyGroupPolyhedra.minimized_constraints()


def computeEightPointsRegion(planeId, edgeId, insideVertex, outsideVertex, xpixel, ypixel, pd, mxp, myp, mzp, ix, iy, iz, mp, mq,
                              posXp, posYp, posZp):
    
    x1 = vertices[outsideVertex*3 + 0]
    y1 = vertices[outsideVertex*3 + 1]
    z1 = vertices[outsideVertex*3 + 2]

    x0 = vertices[insideVertex*3 + 0]
    y0 = vertices[insideVertex*3 + 1]
    z0 = vertices[insideVertex*3 + 2]
    
    
    pointsList = []
    near = 1
    far =1000
    
    pd2 = NNC_Polyhedron(3, 'empty')
    
    
    # print(x0,y0,z0,x1,y1,z1)
    if planeId == 0:
        # print("top plane")
        
        nearRight =0.35820895522388063
        farRight = 358.20895522388063
    
        nearPart = (nearRight*2)/49
        farPart = (farRight*2)/49
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel-24.5)*nearPart
        ypos0 = y0 - nearRight
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x0-(xpixel+1-24.5)*nearPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel-24.5)*nearPart
        ypos0 = y1 - nearRight
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x1-(xpixel+1-24.5)*nearPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel-24.5)*farPart
        ypos0 = y0 - farRight
        zpos0 = z0 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x0-(xpixel+1-24.5)*farPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel-24.5)*farPart
        ypos0 = y1 - farRight
        zpos0 = z1 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x1-(xpixel+1-24.5)*farPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        
    
    if planeId == 1:
        # print("bottom plane")
        
        nearRight =0.35820895522388063
        farRight = 358.20895522388063
        
        nearPart = (nearRight*2)/49
        farPart = (farRight*2)/49
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel-24.5)*nearPart
        ypos0 = y0 + nearRight
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x0-(xpixel+1-24.5)*nearPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel-24.5)*nearPart
        ypos0 = y1 + nearRight
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x1-(xpixel+1-24.5)*nearPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x0-(xpixel-24.5)*farPart
        ypos0 = y0 + farRight
        zpos0 = z0 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x0-(xpixel+1-24.5)*farPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1-(xpixel-24.5)*farPart
        ypos0 = y1 + farRight
        zpos0 = z1 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = x1-(xpixel+1-24.5)*farPart
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
    
    
    if planeId == 2:
        # print("right plane")
        
        nearRight =0.35820895522388063
        farRight = 358.20895522388063
        
        nearPart = (nearRight*2)/49
        farPart = (farRight*2)/49
        
        tempList = [0,0,0]
        xpos0 = x0 - nearRight
        ypos0 = y0+ (ypixel-24.5)*nearPart
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] = y0+(ypixel+1-24.5)*nearPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1 - nearRight
        ypos0 =  y1+(ypixel-24.5)*nearPart
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0                            
        tempList[1] = y1+(ypixel+1-24.5)*nearPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x0 - farRight
        ypos0 = y0+(ypixel-24.5)*farPart        
        zpos0 = z0 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] = y0+(ypixel+1-24.5)*farPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1 - farRight 
        ypos0 = y1+(ypixel-24.5)*farPart
        zpos0 = z1 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] =  y1+(ypixel+1-24.5)*farPart
        tempList[2] = zpos0
        pointsList.append(tempList)
    
    
    if planeId == 3:
        # print("left plane")
        
        nearRight =0.35820895522388063
        farRight = 358.20895522388063
        
        nearPart = (nearRight*2)/49
        farPart = (farRight*2)/49
        
        tempList = [0,0,0]
        xpos0 = x0 + nearRight
        ypos0 = y0+ (ypixel-24.5)*nearPart
        zpos0 = z0 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] = y0+(ypixel+1-24.5)*nearPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1 + nearRight
        ypos0 =  y1+(ypixel-24.5)*nearPart
        zpos0 = z1 + 1
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0                            
        tempList[1] = y1+(ypixel+1-24.5)*nearPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x0 + farRight
        ypos0 = y0+(ypixel-24.5)*farPart        
        zpos0 = z0 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] = y0+(ypixel+1-24.5)*farPart
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        xpos0 = x1 + farRight 
        ypos0 = y1+(ypixel-24.5)*farPart
        zpos0 = z1 + 1000
        tempList[0] = xpos0
        tempList[1] = ypos0
        tempList[2] = zpos0
        pointsList.append(tempList)
        
        tempList = [0,0,0]
        tempList[0] = xpos0
        tempList[1] =  y1+(ypixel+1-24.5)*farPart
        tempList[2] = zpos0
        pointsList.append(tempList)  
        
    
    
    # set_option(rational_to_decimal=True)
    for i in [0,1,3,2,4,5,7,6]:
        
        currPos = pointsList[i]
        # print(currPos)

        # print("currPos")
        mxp1 = currPos[0]
        myp1 = currPos[1]
        mzp1 = currPos[2]

        # mxp1 = str(mxp1).replace("?", "")
        # myp1 = str(myp1).replace("?", "")
        # mzp1 = str(mzp1).replace("?", "")

        ################

        mxp1 = int(float(mxp1)*pow(10, 16))
        myp1 = int(float(myp1)*pow(10, 16))
        mzp1 = int(float(mzp1)*pow(10, 16))

        # print(mxp1,myp1,mzp1)

        pd2.add_generator(point(mxp1*xp0+myp1 * yp0+mzp1*zp0, pow(10, 16)))
        
    # print(pd2.minimized_constraints())
    return pd2
    

def computeRegion(currGroupName, currZP, numberOfFullyInsideVertices, insideVertexDetailsToPPL, numberOfIntersectingEdges,
                  intersectingEdgeDataToPPL, posXp1, posYp1, posZp1, mxp, myp, mzp, outcodeP0, currImageName, currGroupPolyhedra=dummyPolyhedraCons):
    print("\ninside compute region function")
    print("\n\nPPL: Data received")
    print(insideVertexDetailsToPPL)
    print(currImageName)
    # print(insideVertexDetailsToPPL[0][1])
    # if(currImageName == "A_0" and insideVertexDetailsToPPL[0][0] == 94 ):
    #     sleep(5)
    #     insideVertexDetailsToPPL[0][1] =44
    #     insideVertexDetailsToPPL[0][2] =42
    print(insideVertexDetailsToPPL)
    print("intersecting edge data")
    print(intersectingEdgeDataToPPL)
    # print(outcodeP0)
    # exit()
    # print(mxp, myp, mzp, posXp1, posYp1, posZp1)

    mxpOg = mxp
    mypOg = myp
    mzpOg = mzp
    # print("\n")
    imageFrustumPolyhedron = NNC_Polyhedron(3)

    # print("Image frustum polyhedron cons befor adding cons")
    # getCurrentPosOutcodeCons2(outcodeP0, imageFrustumPolyhedron)
    # print(imageFrustumPolyhedron.constraints())
    # print("\n")
    # print(imageFrustumPolyhedron.minimized_constraints())
    # print("\n\n\n")

    currImageCube_ph = NNC_Polyhedron(3)
    
    currImageName.startswith("split")

    if(currImageName != "singlePosImage" and (not currImageName.startswith("split"))):
        print("pos from original computation")
        pathLength = currImageName.count('_')
        currImage_cubeName = ""

        if(pathLength == 1):
            print("image from inital region")
            currImage_cubeName = "initCubeCon"
            # print(currImage_cubeName)

        else:
            print("image from step = ", pathLength)
            currImage_cubeName = currImageName[0:currImageName.rfind("_")]
            #

        # print(currImage_cubeName)

        currImageCube_ph.add_constraints(
            environment.groupCubePostRegion[currImage_cubeName])
        print("currImageCube_ph ==> ", currImageCube_ph.minimized_constraints())

    elif (currImageName == "singlePosImage"):
        # print("single PosImage")
        currImageCube_ph.add_constraints(currGroupPolyhedra)
        print("currImageCube_ph ==> ", currImageCube_ph.minimized_constraints())
    else:
        print("Split region")
        # sleep(2)
        
        
        currImageCube_ph.add_constraints(environment.splitRegionPd["split_"+str(environment.splitCount)])
        print("currImageCube_ph ==> ", currImageCube_ph.minimized_constraints())
        
        # sleep(2)
        
        
        
        

    imageFrustumPolyhedron.intersection_assign(currImageCube_ph)
    print("imageFrustumPolyhedron ==> ",
          imageFrustumPolyhedron.minimized_constraints())

    print("numberOfFullyInsideVertices = ", numberOfFullyInsideVertices)

    for i in range(0, numberOfFullyInsideVertices):

        currentVertexIndex = insideVertexDetailsToPPL[i][0]
        xpixel = insideVertexDetailsToPPL[i][1]
        ypixel = insideVertexDetailsToPPL[i][2]
        # print("\n\n currentVertexIndex :",currentVertexIndex, " xpixel :",xpixel," ypixel : ",ypixel)
        # print(imageFrustumPolyhedron.minimized_constraints())
        addAvertexPixelConstraint(
            currZP, currentVertexIndex, xpixel, ypixel, imageFrustumPolyhedron)
        print(imageFrustumPolyhedron.constraints())
        print(imageFrustumPolyhedron.minimized_constraints())
        
    # for i in range(0, numberOfIntersectingEdges):
    #     xpixel = eval(str(intersectingEdgeDataToPPL[i][4]))
    #     ypixel = eval(str(intersectingEdgeDataToPPL[i][5]))
        
        

    #     ix = eval(str(intersectingEdgeDataToPPL[i][6]).replace("\n",""))
    #     iy = eval(str(intersectingEdgeDataToPPL[i][7]).replace("\n",""))
    #     iz = eval(str(intersectingEdgeDataToPPL[i][8]).replace("\n",""))
    #     planeId = eval(str(intersectingEdgeDataToPPL[i][3]))
    #     print(ix,"\n",iy,"\n",iz)
    #     print(xpixel, ypixel)
    #     # if(planeId == 0 or planeId ==1):
    #     addAvertexPixelConstraintIntersect(currZP,ix,iy,iz, xpixel, ypixel, imageFrustumPolyhedron,planeId)
    #     print(imageFrustumPolyhedron.minimized_constraints())
    #     # sleep(2)

    # addCamerPosCons(posXp,posYp,posZp,imageFrustumPolyhedron)
    # print(imageFrustumPolyhedron.minimized_constraints())

    # print("Fully inside vertices processed ")
    # print(imageFrustumPolyhedron.minimized_constraints())
    # print("\n\n")

    # for i in range(0,numberOfIntersectingEdges):
    # print("Intersecting edges started\n")
    pdC = NNC_Polyhedron(3)
    pdC = imageFrustumPolyhedron

    # print("Number of intersecting edges = ", numberOfIntersectingEdges)
    for i in range(0, numberOfIntersectingEdges):
    # for i in range(0,0):
        # if i!=1:
        #     continue

        edgeId = intersectingEdgeDataToPPL[i][0]
        # if(edgeId ==5 or edgeId ==6 or edgeId ==7):
        #     continue
        # print("Current intersecting edge = ", i, edgeId)
        insideVertex = intersectingEdgeDataToPPL[i][1]
        outsideVertex = intersectingEdgeDataToPPL[i][2]

        planeId = eval(str(intersectingEdgeDataToPPL[i][3]))
        xpixel = eval(str(intersectingEdgeDataToPPL[i][4]))
        ypixel = eval(str(intersectingEdgeDataToPPL[i][5]))

        ix = intersectingEdgeDataToPPL[i][6]
        iy = intersectingEdgeDataToPPL[i][7]
        iz = intersectingEdgeDataToPPL[i][8]

        mp = intersectingEdgeDataToPPL[i][9]
        mq = intersectingEdgeDataToPPL[i][10]

        # if edgeId % 5 == 0:
        #     print(edgeId)
        # print("insideVertex : ",insideVertex," OutsideVertex : ",outsideVertex," xpixel : ",xpixel,"  ypixel: ", ypixel)

        # sleep(4)
        # intersectingEdgeRegion(planeId, edgeId, insideVertex,outsideVertex, xpixel, ypixel, imageFrustumPolyhedron,\
        #     posXp,posYp,posZp)
        
        if planeId == 5:
            continue

        # if(edgeId !=5 and edgeId != 38):
        pd2 = NNC_Polyhedron(3, 'empty')
        
        # if planeId != 2:
        #     continue
        # pd2 = (computeIntersectingRegionUpdated(planeId, edgeId, insideVertex, outsideVertex, xpixel, ypixel, imageFrustumPolyhedron,
        #                                   mxp, myp, mzp, ix, iy, iz, mp, mq, posXp1, posYp1, posZp1))
        
        pd2 = computeEightPointsRegion(planeId, edgeId, insideVertex, outsideVertex, xpixel, ypixel, imageFrustumPolyhedron,
                                          mxp, myp, mzp, ix, iy, iz, mp, mq, posXp1, posYp1, posZp1)
        
        # pd2 = intersectingEdgeRegion(planeId, edgeId, insideVertex,outsideVertex, xpixel, ypixel,imageFrustumPolyhedron )
        # global unsatFlag
        # if(unsatFlag == 1):
        #     unsatFlag = 0
        #     continue
        print(pd2.minimized_constraints())
        
        # if i == 0:
        #     pdC = pd2
        # else:
        #     pdC.intersection_assign(pd2)
        pdC.intersection_assign(pd2)
        # sleep(1)
        # print("\n\n")

    # set_option(rational_to_decimal=False)

    # # print(mxp,myp,mzp,posXp1,posYp1,posZp1)
    # # print(mxp)
    # # print(mxpOg)
    # mxp1 = mxp
    # myp1 = myp
    # mzp1 = mzp
    # mxpStr = [0, 1]
    # mypStr = [0, 1]
    # mzpStr = [0, 1]
    # # Find lcm of denomenators of the pos values
    # if "/" in str(mxp1):
    #     mxpStr = str(mxp1).split("/")
    # else:
    #     mxpStr[0] = str(mxp1)
    #     mxpStr[1] = 1

    # if "/" in str(myp1):
    #     mypStr = str(myp1).split("/")
    # else:
    #     mypStr[0] = str(myp1)
    #     mypStr[1] = 1

    # if "/" in str(mzp1):
    #     mzpStr = str(mzp1).split("/")
    # else:
    #     mzpStr[0] = str(mzp1)
    #     mzpStr[1] = 1

    # numeratorList = [int(str(mxpStr[0])), int(
    #     str(mypStr[0])), int(str(mzpStr[0]))]
    # denomList = [int(int(str(mxpStr[1]))//1),
    #              int(int(str(mypStr[1]))//1), int(int(str(mzpStr[1]))//1)]

    # currLcm = lcm(denomList[0], lcm(denomList[1], denomList[2]))
    # c0 = int(currLcm//(denomList[0]))
    # c1 = int(currLcm//(denomList[1]))
    # c2 = int(currLcm//(denomList[2]))
    # # # # print("\nc0,c1,c2 ", c0,c1,c2)

    # pd100 = NNC_Polyhedron(3, 'empty')
    # xp0 = Variable(0)
    # yp0 = Variable(1)
    # zp0 = Variable(2)

    # # pd100.add_generator(point( mxp1*xp0+myp1* yp0+mzp1*zp0 ,pow(10,80)))

    # pd100.add_generator(point(
    #     numeratorList[0]*c0 * xp0+numeratorList[1] * c1 * yp0+numeratorList[2] * c2*zp0, int(currLcm)))

    # pd100.add_generator(point(
    #     numeratorList[0]*c0 * (xp0+1)+numeratorList[1] * c1 * yp0+numeratorList[2] * c2*zp0, int(currLcm)))

    # pd100.add_generator(point(
    #     numeratorList[0]*c0 * (xp0-1)+numeratorList[1] * c1 * yp0+numeratorList[2] * c2*zp0, int(currLcm)))
    
    # pd100.add_generator(point(
    #     numeratorList[0]*c0 * xp0+numeratorList[1] * c1 * (yp0+1)+numeratorList[2] * c2*zp0, int(currLcm)))

    # pd100.add_generator(point(
    #     numeratorList[0]*c0 * xp0+numeratorList[1] * c1 * (yp0-1)+numeratorList[2] * c2*zp0, int(currLcm)))
    
    # pd100.add_generator(point(
    #     numeratorList[0]*c0 * xp0+numeratorList[1] * c1 * (yp0)+numeratorList[2] * c2*(zp0+1), int(currLcm)))
    
    # pd100.add_generator(point(
    #     numeratorList[0]*c0 * xp0+numeratorList[1] * c1 * (yp0)+numeratorList[2] * c2*(zp0-1), int(currLcm)))





    # print("\n pd100.minimized_constraints()")
    # print( pd100.minimized_constraints())

    # print("\n\n (pdC.minimized_constraints()) before mpos add" )
    # print(pdC.minimized_constraints())

    ################################################
    # pdC.poly_hull_assign(pd100)
    ##################################################
    
    
    # print("\n\npdC.minimized_constraints() after add")
    # print(pdC.minimized_constraints())

    # sleep(3)

    # print("\nFinal pd cons :")
    # print(imageFrustumPolyhedron.minimized_constraints())
    # return str(imageFrustumPolyhedron.minimized_constraints())
    
    
    # for i in range(0, numberOfIntersectingEdges):
    #     xpixel = eval(str(intersectingEdgeDataToPPL[i][4]))
    #     ypixel = eval(str(intersectingEdgeDataToPPL[i][5]))
        
        

    #     ix = eval(str(intersectingEdgeDataToPPL[i][6]).replace("\n",""))
    #     iy = eval(str(intersectingEdgeDataToPPL[i][7]).replace("\n",""))
    #     iz = eval(str(intersectingEdgeDataToPPL[i][8]).replace("\n",""))
    #     planeId = eval(str(intersectingEdgeDataToPPL[i][3]))
    #     print(ix,"\n",iy,"\n",iz)
    #     print(xpixel, ypixel)
    #     # if(planeId == 0 or planeId ==1):
    #     addAvertexPixelConstraintIntersect(currZP,ix,iy,iz, xpixel, ypixel, pdC,planeId)
    #     print(imageFrustumPolyhedron.minimized_constraints())
    #     # sleep(2)

    print(pdC.minimized_constraints())

    environment.imageCons[currImageName] = pdC.minimized_constraints()
    # return str(pdC.minimized_constraints())
    return pdC
