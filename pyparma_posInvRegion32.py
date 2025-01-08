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
   
    # a = np.asarray(a, dtype='float64')
    # b = np.asarray(b, dtype='float64')
    # print(a,b)
    try:
        return np.lcm(a, b)
    except:
        
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


def addAvertexPixelConstraint(currZP, vertexNumber, pixelX, pixelY, pd):
    x = vertices[vertexNumber*3+0]
    y = vertices[vertexNumber*3+1]
    z = vertices[vertexNumber*3+2]
    if (isinstance(x, float)):
       
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
        pd.add_constraint(((-67*qz*(px - qx*xp0)) + (24*qx *
                          (pz - qz*zp0)*1)) <= ((pixelX)*qx*(pz - qz*zp0)*1))
        pd.add_constraint(((-67*qz*(px - qx*xp0)) + (24*qx *
                          (pz - qz*zp0)*1)) > (((pixelX+1))*qx*(pz - qz*zp0)*1))

        pd.add_constraint(((67*qz*(py - qy*yp0)) + (24*qy *
                          (pz - qz*zp0)*1)) <= ((pixelY)*qy*(pz - qz*zp0)*1))
        pd.add_constraint(((67*qz*(py - qy*yp0)) + (24*qy *
                          (pz - qz*zp0)*1)) > (((pixelY+1))*qy*(pz - qz*zp0)*1))

    else:
        pd.add_constraint(((-67*qz*(px - qx*xp0)) + (24*qx *
                          (pz - qz*zp0)*1)) >= ((pixelX)*qx*(pz - qz*zp0)*1))
        pd.add_constraint(((-67*qz*(px - qx*xp0)) + (24*qx *
                          (pz - qz*zp0)*1)) < (((pixelX+1))*qx*(pz - qz*zp0)*1))

        pd.add_constraint(((67*qz*(py - qy*yp0)) + (24*qy *
                          (pz - qz*zp0)*1)) >= ((pixelY)*qy*(pz - qz*zp0)*1))
        pd.add_constraint(((67*qz*(py - qy*yp0)) + (24*qy *
                          (pz - qz*zp0)*1)) < (((pixelY+1))*qy*(pz - qz*zp0)*1))


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
        
        x = vertices[i*3+0]
        y = vertices[i*3+1]
        z = vertices[i*3+2]
        
        if (isinstance(x, float)):
            xf = str(Fraction(x).limit_denominator())
            xl = xf.split('/')
           
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


def findPos(x0, y0, z0, zpos, xpixel, ypixel, planeId):

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

    

    if planeId == 0:
        # print("top plane")
        plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, 0.35820895522388063, -1]
        plane0_v2 = [-358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, 358.20895522388063, -1000]
        cons1 = "xpixel == ((-67*(x0-xp))/(z0-zpos))+24"
        if xpixel < 0:
            xpixel = 0
        elif xpixel > 48:
            xpixel = 48.9
        cons1 = "xpixel == ((-67*(x0-xp))/(z0-zpos))+24"
        if ypixel != 0:
            # print("ypixel not zero")
            cons2 = "0-b == ((67*(y0-yp))/(z0-zpos))+24"
        else:
            cons2 = "ypixel == ((67*(y0-yp))/(z0-zpos))+24"

    if planeId == 1:
        # print("bottom plane")
        plane0_v0 = [-0.35820895522388063, -0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [-358.20895522388063, -358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, -358.20895522388063, -1000]
        if xpixel < 0:
            xpixel = 0
        elif xpixel > 48:
            xpixel = 48.9
        cons1 = "xpixel == ((-67*(x0-xp))/(z0-zpos))+24"
        # cons2 = "48-b == ((67*(y0-yp))/(z0-zpos))+24"
        if ypixel != 48:
            # print("ypixel not 48")
            cons2 = "48.00000000000003-b == ((67*(y0-yp))/(z0-zpos))+24"
        else:
            cons2 = "ypixel == ((67*(y0-yp))/(z0-zpos))+24"

    if planeId == 2:
        # print("right plane")
        plane0_v0 = [0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [358.20895522388063, -358.20895522388063, -1000]
        # cons1 = "48-b == ((-67*(x0-xp))/(z0-zpos))+24"
        if xpixel != 48:
            cons1 = "48.00000000000003-b == ((-67*(x0-xp))/(z0-zpos))+24"
        else:
            cons1 = "xpixel == ((-67*(x0-xp))/(z0-zpos))+24"
        if ypixel > 48:
            #print("ypixel >48")
            ypixel = 48.9
        elif ypixel < 0:
            ypixel = 0
        cons2 = "ypixel == ((67*(y0-yp))/(z0-zpos))+24"
    if planeId == 3:
        # print("left plane ")
        plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [-0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [-358.20895522388063, 358.20895522388063, -1000]
        plane0_v3 = [-358.20895522388063, -358.20895522388063, -1000]
        # cons1 = "0-b == ((-67*(x0-xp))/(z0-zpos))+24"
        if xpixel != 0:
            # print("xpixel not zero")
            cons1 = "0-b == ((-67*(x0-xp))/(z0-zpos))+24"
        else:
            cons1 = "xpixel == ((-67*(x0-xp))/(z0-zpos))+24"
        if ypixel > 48:
            #print("ypixel >48")
            ypixel = 48.9
        elif ypixel < 0:
            ypixel = 0
        cons2 = "ypixel == ((67*(y0-yp))/(z0-zpos))+24"
    if planeId == 5:
        plane0_v0 = [-0.35820895522388063, 0.35820895522388063, -1]
        plane0_v1 = [-0.35820895522388063, -0.35820895522388063, -1]
        plane0_v2 = [0.35820895522388063, -0.35820895522388063, -1]
        plane0_v3 = [0.35820895522388063, 0.35820895522388063, -1]
        cons1 = "xpixel == ((-67*(x0-xp))/(z0-zpos))+24"
        cons2 = "ypixel == ((67*(y0-yp))/(z0-zpos))+24"

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

    s.add(x0-xp == (u*px0+v*px1+w*px2+g*px3))
    s.add(y0-yp == (u*py0+v*py1+w*py2+g*py3))
    s.add(z0-zpos == (u*pz0+v*pz1+w*pz2+g*pz3))

    # print()
    s.check()
    # m= s.model()
    # print(m)

    # cons1 = "xpixel == ((-67*(x0-xp))/(z0-zpos))+24"
    # cons2 = "ypixel == ((67*(y0-yp))/(z0-zpos))+24"

    # print(eval(cons1))
    # print(eval(cons2))
    s.add(eval(cons1))
    s.add(eval(cons2))
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
       
        del(s)
        sleep(10)
        findPos(x0, y0, z0, zpos, xpixel, ypixel, planeId)
    else:
       
        del(s)
        global unsatFlag
        unsatFlag = 1
        exit()
        return 0


# -6699932362587579


def computeIntersectingRegion(planeId, edgeId, insideVertex, outsideVertex, xpixel, ypixel, pd, mxp, myp, mzp, ix, iy, iz, mp, mq,
                              posXp, posYp, posZp):

    x0 = vertices[outsideVertex*3 + 0]
    y0 = vertices[outsideVertex*3 + 1]
    z0 = vertices[outsideVertex*3 + 2]

    x1 = vertices[insideVertex*3 + 0]
    y1 = vertices[insideVertex*3 + 1]
    z1 = vertices[insideVertex*3 + 2]

    # insideFraction = eval("m[p].numerator_as_long()/m[p].denominator_as_long() " )
    # outsideFraction = eval("m[q].numerator_as_long()/m[q].denominator_as_long() " )

    # binsideFraction = 1

    # if(insideFraction+.2 >=1):

    zpos = (eval("mzp.numerator_as_long()/mzp.denominator_as_long()"))

    z0Backwardpos = z0+1000
    # print("z0Backwardpos : ",z0Backwardpos)
    z0Forwardpos = z0+1

    z1Backwardpos = z1+1000
    z1Forwardpos = z1+1

    # if(z0Backwardpos<z0+1):
    #     z0Backwardpos = z0+1

    # if(z1Backwardpos<z1+1):
    #     z1Backwardpos = z1+1

    # if(z0Forwardpos<z0+1):
    #     z0Forwardpos = z0+1

    # if(z1Forwardpos<z1+1):
    #     z1Forwardpos = z1+1.1

    pointsList = []
    # tempList = [posXp,posYp,posZp]
    # pointsList.append(tempList)

    if planeId == 0 or planeId == 1:
        # top or bottom plane
        print("top/bottom plane")
        # if(planeId == 0):
        #     ypixel =0
        # else:
        #     ypixel = 48

        xpixel = math.floor(float(xpixel))

        # tempList =findPos(ix,iy,iz,zpos,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPos(ix,iy,iz,zpos,xpixel+1,ypixel,planeId)
        # pointsList.append(tempList)
        tempList = findPos(x0, y0, z0, z0Backwardpos, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPos(x0, y0, z0, z0Backwardpos,
                           xpixel+1, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPos(x0, y0, z0, z0Forwardpos, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPos(x0, y0, z0, z0Forwardpos, xpixel+1, ypixel, planeId)
        pointsList.append(tempList)

        # tempList =findPos(ix,iy,iz,zpos,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPos(ix,iy,iz,zpos,xpixel+1,ypixel,planeId)
        # pointsList.append(tempList)
        tempList = findPos(x1, y1, z1, z1Backwardpos, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPos(x1, y1, z1, z1Backwardpos,
                           xpixel+1, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPos(x1, y1, z1, z1Forwardpos, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPos(x1, y1, z1, z1Forwardpos, xpixel+1, ypixel, planeId)
        pointsList.append(tempList)

    if planeId == 2 or planeId == 3:
        print("left/right plane")
        # if(planeId == 2):
        #     xpixel =48
        # else:
        #     xpixel = 0

        ypixel = math.floor(float(ypixel))

        # tempList =findPos(ix,iy,iz,zpos,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPos(ix,iy,iz,zpos,xpixel,ypixel+1,planeId)
        # pointsList.append(tempList)
        tempList = findPos(x0, y0, z0, z0Backwardpos, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPos(x0, y0, z0, z0Backwardpos,
                           xpixel, ypixel+1, planeId)
        pointsList.append(tempList)
        tempList = findPos(x0, y0, z0, z0Forwardpos, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPos(x0, y0, z0, z0Forwardpos, xpixel, ypixel+1, planeId)
        pointsList.append(tempList)

        # tempList =findPos(ix,iy,iz,zpos,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPos(ix,iy,iz,zpos,xpixel,ypixel+1,planeId)
        # pointsList.append(tempList)
        tempList = findPos(x1, y1, z1, z1Backwardpos, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPos(x1, y1, z1, z1Backwardpos,
                           xpixel, ypixel+1, planeId)
        pointsList.append(tempList)
        tempList = findPos(x1, y1, z1, z1Forwardpos, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPos(x1, y1, z1, z1Forwardpos, xpixel, ypixel+1, planeId)
        pointsList.append(tempList)

        # tempList =findPos(ix,iy,iz,zpos,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPos(ix,iy,iz,zpos,xpixel,ypixel+1,planeId)
        # pointsList.append(tempList)
        # tempList =findPos(x0,y0,z0,zpos-5,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPos(x0,y0,z0,zpos-5,xpixel,ypixel+1,planeId)
        # pointsList.append(tempList)
        # tempList =findPos(x0,y0,z0,zpos+5,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPos(x0,y0,z0,zpos+5,xpixel,ypixel+1,planeId)
        # pointsList.append(tempList)

        # tempList =findPos(ix,iy,iz,zpos,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPos(ix,iy,iz,zpos,xpixel,ypixel+1,planeId)
        # pointsList.append(tempList)
        # tempList =findPos(x1,y1,z1,zpos-5,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPos(x1,y1,z1,zpos-5,xpixel,ypixel+1,planeId)
        # pointsList.append(tempList)
        # tempList =findPos(x1,y1,z1,zpos+5,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPos(x1,y1,z1,zpos+5,xpixel,ypixel+1,planeId)
        # pointsList.append(tempList)
    if planeId == 5 or planeId == 6:

        # print("near far plane pyparma pos inva region 19 ")
        # sleep(10)
        xpixel = math.floor(float(xpixel))
        ypixel = math.floor(float(ypixel))

        # tempList =findPos(ix,iy,iz,zpos,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPos(ix,iy,iz,zpos,xpixel,ypixel+1,planeId)
        # pointsList.append(tempList)
        tempList = findPos(x0, y0, z0, z0Backwardpos, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPos(x0, y0, z0, z0Backwardpos,
                           xpixel, ypixel+1, planeId)
        pointsList.append(tempList)
        tempList = findPos(x0, y0, z0, z0Forwardpos, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPos(x0, y0, z0, z0Forwardpos, xpixel, ypixel+1, planeId)
        pointsList.append(tempList)

        # tempList =findPos(ix,iy,iz,zpos,xpixel,ypixel,planeId)
        # pointsList.append(tempList)
        # tempList =findPos(ix,iy,iz,zpos,xpixel,ypixel+1,planeId)
        # pointsList.append(tempList)
        tempList = findPos(x1, y1, z1, z1Backwardpos, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPos(x1, y1, z1, z1Backwardpos,
                           xpixel, ypixel+1, planeId)
        pointsList.append(tempList)
        tempList = findPos(x1, y1, z1, z1Forwardpos, xpixel, ypixel, planeId)
        pointsList.append(tempList)
        tempList = findPos(x1, y1, z1, z1Forwardpos, xpixel, ypixel+1, planeId)
        pointsList.append(tempList)

    pd2 = NNC_Polyhedron(3, 'empty')
    pd0 = NNC_Polyhedron(3, 'empty')
    pd10 = NNC_Polyhedron(3, 'empty')

    # for i in range(0, 3):
    #     currPos = pointsList[i]
    #     # print(currPos)

    #     xcord = int(float(currPos[0])*pow(10,240)//1)
    #     ycord = int(float(currPos[1])*pow(10,240)//1)
    #     zcord = int(float(currPos[2])*pow(10,240)//1)

    #     # print(xcord, ycord, zcord)
    #     pd0.add_generator(point( xcord*xp0+ycord*yp0+zcord*zp0 ,pow(10,240)))
    #     # print(pd2.generators())

    # print("\npd0 minimized_constraints")
    # print(pd0.minimized_constraints())

    # for i in range(3, 7):
    #     currPos = pointsList[i]
    #     # print(currPos)

    #     xcord = int(float(currPos[0])*pow(10,240)//1)
    #     ycord = int(float(currPos[1])*pow(10,240)//1)
    #     zcord = int(float(currPos[2])*pow(10,240)//1)

    #     # print(xcord, ycord, zcord)
    #     pd10.add_generator(point( xcord*xp0+ycord*yp0+zcord*zp0 ,pow(10,240)))
    #     # print(pd2.generators())

    # print("\npd10 minimized_constraints")
    # print(pd10.minimized_constraints())

    set_option(rational_to_decimal=True)
    for i in range(0, 8):
        global unsatFlag
        if(unsatFlag == 1):
            return 0
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

        mxp1 = int(float(mxp1)*pow(10, 58))
        myp1 = int(float(myp1)*pow(10, 58))
        mzp1 = int(float(mzp1)*pow(10, 58))

        # print(mxp1,myp1,mzp1)

        pd2.add_generator(point(mxp1*xp0+myp1 * yp0+mzp1*zp0, pow(10, 58)))

    #print(x0, y0, z0,x1,y1,z1)
    # print("\n\nIntersection assign pd, whole invariant region.")
    # print("before intersection")
    # print(pd2.minimized_constraints())
    # pd=pd2
    # pd.intersection_assign(pd2)
    # print(pd.minimized_constraints())
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

   
    sleep(300)
    exit(0)
    return pdIntersection

dummyGroupPolyhedra=NNC_Polyhedron(3)
dummyPolyhedraCons = dummyGroupPolyhedra.minimized_constraints()

def computeRegion(currGroupName, currZP, numberOfFullyInsideVertices, insideVertexDetailsToPPL, numberOfIntersectingEdges,
                  intersectingEdgeDataToPPL, posXp1, posYp1, posZp1, mxp, myp, mzp, outcodeP0, currImageName, currGroupPolyhedra=dummyPolyhedraCons):

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

      

        currImageCube_ph.add_constraints(
            environment.groupCubePostRegion[currImage_cubeName])
        

    elif (currImageName == "singlePosImage"):
        
        currImageCube_ph.add_constraints(currGroupPolyhedra)
       
    else:
       
        
        
        currImageCube_ph.add_constraints(environment.splitRegionPd["split_"+str(environment.splitCount)])
        
        
        # sleep(2)
        
        
        
        

    imageFrustumPolyhedron.intersection_assign(currImageCube_ph)
   
    for i in range(0, numberOfFullyInsideVertices):

        currentVertexIndex = insideVertexDetailsToPPL[i][0]
        xpixel = insideVertexDetailsToPPL[i][1]
        ypixel = insideVertexDetailsToPPL[i][2]
        # print("\n\n currentVertexIndex :",currentVertexIndex, " xpixel :",xpixel," ypixel : ",ypixel)
        # print(imageFrustumPolyhedron.minimized_constraints())
        addAvertexPixelConstraint(
            currZP, currentVertexIndex, xpixel, ypixel, imageFrustumPolyhedron)
        # print(imageFrustumPolyhedron.constraints())
        # print(imageFrustumPolyhedron.minimized_constraints())

    # addCamerPosCons(posXp,posYp,posZp,imageFrustumPolyhedron)
    # print(imageFrustumPolyhedron.minimized_constraints())

   

    # for i in range(0,numberOfIntersectingEdges):
    
    pdC = NNC_Polyhedron(3)
    pdC = imageFrustumPolyhedron

    
    for i in range(0, numberOfIntersectingEdges):
        # for i in range(0,0):

        edgeId = intersectingEdgeDataToPPL[i][0]
        # if(edgeId ==5 or edgeId ==6 or edgeId ==7):
        #     continue
        
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

        if edgeId % 5 == 0:
            print(edgeId)
            # print("insideVertex : ",insideVertex," OutsideVertex : ",outsideVertex," xpixel : ",xpixel,"  ypixel: ", ypixel)

        # intersectingEdgeRegion(planeId, edgeId, insideVertex,outsideVertex, xpixel, ypixel, imageFrustumPolyhedron,\
        #     posXp,posYp,posZp)

        # if(edgeId !=5 and edgeId != 38):
        pd2 = NNC_Polyhedron(3, 'empty')
        pd2 = (computeIntersectingRegion(planeId, edgeId, insideVertex, outsideVertex, xpixel, ypixel, imageFrustumPolyhedron,
                                         mxp, myp, mzp, ix, iy, iz, mp, mq, posXp1, posYp1, posZp1))
        global unsatFlag
        if(unsatFlag == 1):
            unsatFlag = 0
            continue
      
        pdC.intersection_assign(pd2)
        # print("\n\n")

    set_option(rational_to_decimal=False)

    # print(mxp,myp,mzp,posXp1,posYp1,posZp1)
    # print(mxp)
    # print(mxpOg)
    mxp1 = mxp
    myp1 = myp
    mzp1 = mzp
    mxpStr = [0, 1]
    mypStr = [0, 1]
    mzpStr = [0, 1]
    # Find lcm of denomenators of the pos values
    if "/" in str(mxp1):
        mxpStr = str(mxp1).split("/")
    else:
        mxpStr[0] = str(mxp1)
        mxpStr[1] = 1

    if "/" in str(myp1):
        mypStr = str(myp1).split("/")
    else:
        mypStr[0] = str(myp1)
        mypStr[1] = 1

    if "/" in str(mzp1):
        mzpStr = str(mzp1).split("/")
    else:
        mzpStr[0] = str(mzp1)
        mzpStr[1] = 1

    numeratorList = [int(str(mxpStr[0])), int(
        str(mypStr[0])), int(str(mzpStr[0]))]
    denomList = [int(int(str(mxpStr[1]))//1),
                 int(int(str(mypStr[1]))//1), int(int(str(mzpStr[1]))//1)]

    currLcm = lcm(denomList[0], lcm(denomList[1], denomList[2]))
    c0 = int(currLcm//(denomList[0]))
    c1 = int(currLcm//(denomList[1]))
    c2 = int(currLcm//(denomList[2]))
    # # # print("\nc0,c1,c2 ", c0,c1,c2)

    pd100 = NNC_Polyhedron(3, 'empty')
    xp0 = Variable(0)
    yp0 = Variable(1)
    zp0 = Variable(2)

    # pd100.add_generator(point( mxp1*xp0+myp1* yp0+mzp1*zp0 ,pow(10,80)))

    pd100.add_generator(point(
        numeratorList[0]*c0 * xp0+numeratorList[1] * c1 * yp0+numeratorList[2] * c2*zp0, int(currLcm)))

    # print("\n pd100.minimized_constraints()")
    # print( pd100.minimized_constraints())

    # print("\n\n (pdC.minimized_constraints()) before mpos add" )
    # print(pdC.minimized_constraints())

    pdC.poly_hull_assign(pd100)
    # print("\n\npdC.minimized_constraints() after add")
    # print(pdC.minimized_constraints())

    # sleep(3)

   

    environment.imageCons[currImageName] = pdC.minimized_constraints()
    # return str(pdC.minimized_constraints())
    return pdC
