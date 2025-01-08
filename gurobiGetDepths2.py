#!/usr/bin/env python3.7


import gurobipy as gp
from gurobipy import GRB
import math
import environment


from time import sleep
from decimal import Decimal

# Create a new model
# m = gp.Model("qcp")



#new function to compute min depth using constraints


vertices = environment.vertices

def getDepthIntervals3(currImageSetConString,dataToComputeDepth,
                      currGroupRegionCons,edgeVertexIndices, currTriangleVertices):
    
    print("inside getDepthIntervals3")
    print("currImageSetConString = ", currImageSetConString)
    print("dataToComputeDepth = ", dataToComputeDepth)
    print("currGroupRegionCons = ", currGroupRegionCons)
    print("edgeVertexIndices = ", edgeVertexIndices)
    print("currTriangleVertices = ", currTriangleVertices)
    
    currFrustumPlane = dataToComputeDepth[3]
    print("currFrustumPlane = ", currFrustumPlane)
    insideVertex = dataToComputeDepth[1]
    outsideVertex = dataToComputeDepth[2]
    print("insideVertex = ", insideVertex)
    print("outsideVertex = ", outsideVertex)
    
    xpixel = dataToComputeDepth[4]
    ypixel = dataToComputeDepth[5]
    
    print("xpixel, ypixel = ", xpixel, ypixel)
    
    
    
   
    
    env = gp.Env(empty=True)
    # env.setParam("WLSAccessID", str)
    # env.setParam("WLSSECRET", str)
    # env.setParam("LICENSEID", int)
    env.setParam("OutputFlag", 0)
    env.start()
    
    
    m = gp.Model( "model1", env=env)
    xp0 = m.addVar(lb=-GRB.INFINITY, name="xp0")
    yp0 = m.addVar(lb=-GRB.INFINITY, name="yp0")
    zp0 = m.addVar(lb=-GRB.INFINITY, name="zp0")
    m.setParam(GRB.Param.NonConvex, 2)
    m.setParam(GRB.Param.NumericFocus, 3)
    # # # m.setParam(GRB.Param.OutputFlag, 0)
    m.setParam('BarHomogeneous',1)
    # m.setParam('Method',5)
    
    
    
    # obj = zp0
    
    consList = currImageSetConString.split(",")
    # consList = currImageSetConString
    # print(consList)
    
    for i in range(0, len(consList)):
        consName = "q_"+str(i)
        # print(consName, " :", consList[i])
        currCons = consList[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            # print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        except:
            print("error occured")
            exit(0)
        # m.addConstr(currCons, "str(consName)")

    currRegionConsString = str(currGroupRegionCons)
    currRegionConsString = currRegionConsString.replace("And(","")
    currRegionConsString = currRegionConsString.replace(")","")
    currRegionConsString = currRegionConsString.replace("\n","")
    currRegionConsString = currRegionConsString.replace("  ","")
    print("currRegionConsString = ", currRegionConsString)
    
    regionConsList = currRegionConsString.split(",")
    print("regionConsList = ", regionConsList)
    print(regionConsList[0])
    for i in range(0, len(regionConsList)):
        consName = "qr_"+str(i)
        # print(consName, " :", consList[i])
        currCons = regionConsList[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            # print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        except:
            print("error occured")
            exit(0)
        # m.addConstr(currCons, "str(consName)")
        
    
    print("Adding plane constraints")
    #define the plane constraints
    xf0 = m.addVar(lb=-GRB.INFINITY, name="xf0")
    yf0 = m.addVar(lb=-GRB.INFINITY, name="yf0")
    zf0 = m.addVar(lb=-GRB.INFINITY, name="zf0")
    
    u = m.addVar(lb=-GRB.INFINITY, name="u")
    v = m.addVar(lb=-GRB.INFINITY, name="v")
    w = m.addVar(lb=-GRB.INFINITY, name="w")
    g = m.addVar(lb=-GRB.INFINITY, name="g")
    
    frustumPlaneCons = ["u>=0", "v>=0", "w>=0", "g>=0", "u+v+w+g ==1"]
    
    # plane0_v0 = [-0.35820895522388063,0.35820895522388063,-1]
    # plane0_v1 = [0.35820895522388063,0.35820895522388063,-1]
    # plane0_v2 = [-358.20895522388063,358.20895522388063,-1000]    
    # plane0_v3 = [358.20895522388063,358.20895522388063,-1000] 
    
    
    nTL =["(xp0-0.35820895522388063)", "(yp0+0.35820895522388063)", "(zp0-1)"]
    nTR =["(xp0+0.35820895522388063)", "(yp0+0.35820895522388063)", "(zp0-1)"]
    nBL =["(xp0-0.35820895522388063)", "(yp0-0.35820895522388063)", "(zp0-1)"]
    nBR =["(xp0+0.35820895522388063)", "(yp0-0.35820895522388063)", "(zp0-1)"]


    fTL =["(xp0-358.20895522388063)", "(yp0+358.20895522388063)", "(zp0-1000)"]
    fTR =["(xp0+358.20895522388063)", "(yp0+358.20895522388063)", "(zp0-1000)"]
    fBL =["(xp0-358.20895522388063)", "(yp0-358.20895522388063)", "(zp0-1000)"]
    fBR =["(xp0+358.20895522388063)", "(yp0-358.20895522388063)", "(zp0-1000)"]
    
    
    if currFrustumPlane == 0:
        frustumPlaneCons.append("u*"+str(nTL[0])+"+v*"+str(nTR[0])+"+w*"+str(fTL[0])+"+g*"+str(fTR[0])+" == xf0")
        frustumPlaneCons.append("u*"+str(nTL[1])+"+v*"+str(nTR[1])+"+w*"+str(fTL[1])+"+g*"+str(fTR[1])+" == yf0")
        frustumPlaneCons.append("u*"+str(nTL[2])+"+v*"+str(nTR[2])+"+w*"+str(fTL[2])+"+g*"+str(fTR[2])+" == zf0")
    elif currFrustumPlane == 1:
        #bottom plane
        frustumPlaneCons.append("u*"+str(nBL[0])+"+v*"+str(nBR[0])+"+w*"+str(fBL[0])+"+g*"+str(fBR[0])+" == xf0")
        frustumPlaneCons.append("u*"+str(nBL[1])+"+v*"+str(nBR[1])+"+w*"+str(fBL[1])+"+g*"+str(fBR[1])+" == yf0")
        frustumPlaneCons.append("u*"+str(nBL[2])+"+v*"+str(nBR[2])+"+w*"+str(fBL[2])+"+g*"+str(fBR[2])+" == zf0")
    elif currFrustumPlane == 2:
        #right plane
        frustumPlaneCons.append("u*"+str(nTR[0])+"+v*"+str(fTR[0])+"+w*"+str(fBR[0])+"+g*"+str(nBR[0])+" == xf0")
        frustumPlaneCons.append("u*"+str(nTR[1])+"+v*"+str(fTR[1])+"+w*"+str(fBR[1])+"+g*"+str(nBR[1])+" == yf0")
        frustumPlaneCons.append("u*"+str(nTR[2])+"+v*"+str(fTR[2])+"+w*"+str(fBR[2])+"+g*"+str(nBR[2])+" == zf0")
    elif currFrustumPlane == 3:
        #left plane
        frustumPlaneCons.append("u*"+str(nTL[0])+"+v*"+str(fTL[0])+"+w*"+str(fBL[0])+"+g*"+str(nBL[0])+" == xf0")
        frustumPlaneCons.append("u*"+str(nTL[1])+"+v*"+str(fTL[1])+"+w*"+str(fBL[1])+"+g*"+str(nBL[1])+" == yf0")
        frustumPlaneCons.append("u*"+str(nTL[2])+"+v*"+str(fTL[2])+"+w*"+str(fBL[2])+"+g*"+str(nBL[2])+" == zf0")
    
    
    
    for i in range(0, len(frustumPlaneCons)):
        consName = "qf_"+str(i)
        print(consName, " :", frustumPlaneCons[i])
        currCons = frustumPlaneCons[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            # print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        except:
            print("error occured")
            exit(0)
    
    #add points on the edge constraints
    xl = m.addVar(lb=-GRB.INFINITY, name="xl")
    yl = m.addVar(lb=-GRB.INFINITY, name="yl")
    zl = m.addVar(lb=-GRB.INFINITY, name="zl")
    
    p = m.addVar(lb=-GRB.INFINITY, name="p")
    q = m.addVar(lb=-GRB.INFINITY, name="q")
    
    
    
    
    
    
    lineConsList = ["p>=0", "q>=0", "p+q == 1"]
    lineConsList.append("p*"+str(vertices[insideVertex*3+0])+"+q*"+str(vertices[outsideVertex*3+0])+" == xl")
    lineConsList.append("p*"+str(vertices[insideVertex*3+1])+"+q*"+str(vertices[outsideVertex*3+1])+" == yl")
    lineConsList.append("p*"+str(vertices[insideVertex*3+2])+"+q*"+str(vertices[outsideVertex*3+2])+" == zl")
    
    for i in range(0, len(lineConsList)):
        consName = "qx_"+str(i)
        print(consName, " :", lineConsList[i])
        currCons = lineConsList[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            # print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        except:
            print("error occured")
            exit(0)
   
    
    #add intersecting points constraints
    
    intersectingPointConsList = ["xf0 == xl", "yf0 == yl", "zf0 == zl"]
    for i in range(0, len(intersectingPointConsList)):
        consName = "qfx_"+str(i)
        print(consName, " :", intersectingPointConsList[i])
        currCons = intersectingPointConsList[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            # print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        except:
            print("error occured")
            exit(0)
    
    #add pixel constraints
    
    pixelConsList = []
    
    # pixelConsList = ["-68.39567*(p*("+str(vertices[insideVertex*3+0])+"-xp0)+q*("+str(vertices[outsideVertex*3+0])+" -xp0))  <= (42 -24.5)*(p*("+str(vertices[insideVertex*3+2])+"-zp0)+q*("+str(vertices[outsideVertex*3+2])+" -zp0) )"]
    
    
    pixelConsList = ["-68.39567*(xl-xp0) <= ("+ str(xpixel)+"-24.5)*(zl-zp0)", "68.39567*(yl-yp0) <= ("+ str(ypixel)+"-24.5)*(zl-zp0)"]
    
    # a = -68.39567*((0.10932812550043353356*6.204480171203613+0.89067187449956646643*6.143178939819336) -121.5) 
    # b =  (0.10932812550043353356*92.03382110595703+0.89067187449956646643*99.4977798461914) - 121.5
    # print(a)
    # print(b)
    # print(a/b)
    
    
    
    # pixelConsList.append("68.39567*(yl -yp0)  >= (49-24.5)*(zl -zp0) ")
    for i in range(0, len(pixelConsList)):
        consName = "qp_"+str(i)
        print(consName, " :", pixelConsList[i])
        currCons = pixelConsList[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            # print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        except Exception as e:
            print(str(e))
            print("error occured")
            exit(0)
    obj =  ((xp0 - xl)**2 + (yp0 - yl)**2 + (zp0 - zl)**2)
    print("minimizing...")
    m.setObjective(obj, GRB.MINIMIZE)
    # m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()
    print(m.Status)
    
    print("p = ", p.X)
    print("q = ", q.X)
    
    
    print("xl = ",xl.X)
    print("yl = ",yl.X)
    print("zl = ",zl.X)
    
    print("xp0 = ",xp0.X)
    print("yp0 = ",yp0.X)
    print("zp0 = ",zp0.X)
    
    
    minVal = obj.getValue()
    
    
    
    print("minVal = ", minVal)
    return minVal
                                                                  


def getDepthIntervals4(currImageSetConString,dataToComputeDepth,
                      currGroupRegionCons,edgeVertexIndices, currTriangleVertices):
    
    print("inside getDepthIntervals3")
    print("currImageSetConString = ", currImageSetConString)
    print("dataToComputeDepth = ", dataToComputeDepth)
    print("currGroupRegionCons = ", currGroupRegionCons)
    print("edgeVertexIndices = ", edgeVertexIndices)
    print("currTriangleVertices = ", currTriangleVertices)
    
    currFrustumPlane = dataToComputeDepth[3]
    print("currFrustumPlane = ", currFrustumPlane)
    insideVertex = dataToComputeDepth[1]
    outsideVertex = dataToComputeDepth[2]
    print("insideVertex = ", insideVertex)
    print("outsideVertex = ", outsideVertex)
    
    xpixel = dataToComputeDepth[4]
    ypixel = dataToComputeDepth[5]
    
    print("xpixel, ypixel = ", xpixel, ypixel)
    
    
    
   
    
    env = gp.Env(empty=True)
    # env.setParam("WLSAccessID", str)
    # env.setParam("WLSSECRET", str)
    # env.setParam("LICENSEID", int)
    env.setParam("OutputFlag", 0)
    env.start()
    
    
    m = gp.Model( "model1", env=env)
    xp0 = m.addVar(lb=-GRB.INFINITY, name="xp0")
    yp0 = m.addVar(lb=-GRB.INFINITY, name="yp0")
    zp0 = m.addVar(lb=-GRB.INFINITY, name="zp0")
    m.setParam(GRB.Param.NonConvex, 2)
    m.setParam(GRB.Param.NumericFocus, 3)
    # # # m.setParam(GRB.Param.OutputFlag, 0)
    m.setParam('BarHomogeneous',1)
    # m.setParam('Method',5)
    
    
    
    
    
    consList = currImageSetConString.split(",")
    # consList = currImageSetConString
    # print(consList)
    
    for i in range(0, len(consList)):
        consName = "q_"+str(i)
        # print(consName, " :", consList[i])
        currCons = consList[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            # print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        except:
            print("error occured")
            exit(0)
        # m.addConstr(currCons, "str(consName)")

    currRegionConsString = str(currGroupRegionCons)
    currRegionConsString = currRegionConsString.replace("And(","")
    currRegionConsString = currRegionConsString.replace(")","")
    currRegionConsString = currRegionConsString.replace("\n","")
    currRegionConsString = currRegionConsString.replace("  ","")
    print("currRegionConsString = ", currRegionConsString)
    
    regionConsList = currRegionConsString.split(",")
    print("regionConsList = ", regionConsList)
    print(regionConsList[0])
    for i in range(0, len(regionConsList)):
        consName = "qr_"+str(i)
        # print(consName, " :", consList[i])
        currCons = regionConsList[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            # print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        except:
            print("error occured")
            exit(0)
        # m.addConstr(currCons, "str(consName)")
        
    
    print("Adding plane constraints")
    #define the plane constraints
    xf0 = m.addVar(lb=-GRB.INFINITY, name="xf0")
    yf0 = m.addVar(lb=-GRB.INFINITY, name="yf0")
    zf0 = m.addVar(lb=-GRB.INFINITY, name="zf0")
    
    u = m.addVar(lb=-GRB.INFINITY, name="u")
    v = m.addVar(lb=-GRB.INFINITY, name="v")
    w = m.addVar(lb=-GRB.INFINITY, name="w")
    g = m.addVar(lb=-GRB.INFINITY, name="g")
    
    frustumPlaneCons = ["u>=0", "v>=0", "w>=0", "g>=0", "u+v+w+g ==1"]
    
    # plane0_v0 = [-0.35820895522388063,0.35820895522388063,-1]
    # plane0_v1 = [0.35820895522388063,0.35820895522388063,-1]
    # plane0_v2 = [-358.20895522388063,358.20895522388063,-1000]    
    # plane0_v3 = [358.20895522388063,358.20895522388063,-1000] 
    
    
    nTL =["(xp0-0.35820895522388063)", "(yp0+0.35820895522388063)", "(zp0-1)"]
    nTR =["(xp0+0.35820895522388063)", "(yp0+0.35820895522388063)", "(zp0-1)"]
    nBL =["(xp0-0.35820895522388063)", "(yp0-0.35820895522388063)", "(zp0-1)"]
    nBR =["(xp0+0.35820895522388063)", "(yp0-0.35820895522388063)", "(zp0-1)"]


    fTL =["(xp0-358.20895522388063)", "(yp0+358.20895522388063)", "(zp0-1000)"]
    fTR =["(xp0+358.20895522388063)", "(yp0+358.20895522388063)", "(zp0-1000)"]
    fBL =["(xp0-358.20895522388063)", "(yp0-358.20895522388063)", "(zp0-1000)"]
    fBR =["(xp0+358.20895522388063)", "(yp0-358.20895522388063)", "(zp0-1000)"]
    
    
    if currFrustumPlane == 0:
        frustumPlaneCons.append("u*"+str(nTL[0])+"+v*"+str(nTR[0])+"+w*"+str(fTL[0])+"+g*"+str(fTR[0])+" == xf0")
        frustumPlaneCons.append("u*"+str(nTL[1])+"+v*"+str(nTR[1])+"+w*"+str(fTL[1])+"+g*"+str(fTR[1])+" == yf0")
        frustumPlaneCons.append("u*"+str(nTL[2])+"+v*"+str(nTR[2])+"+w*"+str(fTL[2])+"+g*"+str(fTR[2])+" == zf0")
    elif currFrustumPlane == 1:
        #bottom plane
        frustumPlaneCons.append("u*"+str(nBL[0])+"+v*"+str(nBR[0])+"+w*"+str(fBL[0])+"+g*"+str(fBR[0])+" == xf0")
        frustumPlaneCons.append("u*"+str(nBL[1])+"+v*"+str(nBR[1])+"+w*"+str(fBL[1])+"+g*"+str(fBR[1])+" == yf0")
        frustumPlaneCons.append("u*"+str(nBL[2])+"+v*"+str(nBR[2])+"+w*"+str(fBL[2])+"+g*"+str(fBR[2])+" == zf0")
    elif currFrustumPlane == 2:
        #right plane
        frustumPlaneCons.append("u*"+str(nTR[0])+"+v*"+str(fTR[0])+"+w*"+str(fBR[0])+"+g*"+str(nBR[0])+" == xf0")
        frustumPlaneCons.append("u*"+str(nTR[1])+"+v*"+str(fTR[1])+"+w*"+str(fBR[1])+"+g*"+str(nBR[1])+" == yf0")
        frustumPlaneCons.append("u*"+str(nTR[2])+"+v*"+str(fTR[2])+"+w*"+str(fBR[2])+"+g*"+str(nBR[2])+" == zf0")
    elif currFrustumPlane == 3:
        #left plane
        frustumPlaneCons.append("u*"+str(nTL[0])+"+v*"+str(fTL[0])+"+w*"+str(fBL[0])+"+g*"+str(nBL[0])+" == xf0")
        frustumPlaneCons.append("u*"+str(nTL[1])+"+v*"+str(fTL[1])+"+w*"+str(fBL[1])+"+g*"+str(nBL[1])+" == yf0")
        frustumPlaneCons.append("u*"+str(nTL[2])+"+v*"+str(fTL[2])+"+w*"+str(fBL[2])+"+g*"+str(nBL[2])+" == zf0")
    
    
    
    for i in range(0, len(frustumPlaneCons)):
        consName = "qf_"+str(i)
        print(consName, " :", frustumPlaneCons[i])
        currCons = frustumPlaneCons[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            # print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        except:
            print("error occured")
            exit(0)
    
    #add points on the edge constraints
    xl = m.addVar(lb=-GRB.INFINITY, name="xl")
    yl = m.addVar(lb=-GRB.INFINITY, name="yl")
    zl = m.addVar(lb=-GRB.INFINITY, name="zl")
    
    p = m.addVar(lb=-GRB.INFINITY, name="p")
    q = m.addVar(lb=-GRB.INFINITY, name="q")
    
    
    
    
    
    
    lineConsList = ["p>=0", "q>=0", "p+q == 1"]
    lineConsList.append("p*"+str(vertices[insideVertex*3+0])+"+q*"+str(vertices[outsideVertex*3+0])+" == xl")
    lineConsList.append("p*"+str(vertices[insideVertex*3+1])+"+q*"+str(vertices[outsideVertex*3+1])+" == yl")
    lineConsList.append("p*"+str(vertices[insideVertex*3+2])+"+q*"+str(vertices[outsideVertex*3+2])+" == zl")
    
    for i in range(0, len(lineConsList)):
        consName = "qx_"+str(i)
        print(consName, " :", lineConsList[i])
        currCons = lineConsList[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            # print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        except:
            print("error occured")
            exit(0)
   
    
    #add intersecting points constraints
    
    intersectingPointConsList = ["xf0 == xl", "yf0 == yl", "zf0 == zl"]
    for i in range(0, len(intersectingPointConsList)):
        consName = "qfx_"+str(i)
        print(consName, " :", intersectingPointConsList[i])
        currCons = intersectingPointConsList[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            # print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        except:
            print("error occured")
            exit(0)
    
    #add pixel constraints
    
    pixelConsList = []
    
    # pixelConsList = ["-68.39567*(p*("+str(vertices[insideVertex*3+0])+"-xp0)+q*("+str(vertices[outsideVertex*3+0])+" -xp0))  <= (42 -24.5)*(p*("+str(vertices[insideVertex*3+2])+"-zp0)+q*("+str(vertices[outsideVertex*3+2])+" -zp0) )"]
    
    
    pixelConsList = ["-68.39567*(xl-xp0) <= ("+ str(xpixel)+"-24.5)*(zl-zp0)", "68.39567*(yl-yp0) <= ("+ str(ypixel)+"-24.5)*(zl-zp0)"]
    
    # a = -68.39567*((0.10932812550043353356*6.204480171203613+0.89067187449956646643*6.143178939819336) -121.5) 
    # b =  (0.10932812550043353356*92.03382110595703+0.89067187449956646643*99.4977798461914) - 121.5
    # print(a)
    # print(b)
    # print(a/b)
    
    
    
    # pixelConsList.append("68.39567*(yl -yp0)  >= (49-24.5)*(zl -zp0) ")
    for i in range(0, len(pixelConsList)):
        consName = "qp_"+str(i)
        print(consName, " :", pixelConsList[i])
        currCons = pixelConsList[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            # print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        except Exception as e:
            print(str(e))
            print("error occured")
            exit(0)
    
    obj =  ((xp0 - xl)**2 + (yp0 - yl)**2 + (zp0 - zl)**2)
    # m.setObjective(obj, GRB.MINIMIZE)
    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()
    print(m.Status)
    
    print("p = ", p.X)
    print("q = ", q.X)
    
    
    print("xl = ",xl.X)
    print("yl = ",yl.X)
    print("zl = ",zl.X)
    
    print("xp0 = ",xp0.X)
    print("yp0 = ",yp0.X)
    print("zp0 = ",zp0.X)
    
    
    maxVal = obj.getValue()
    
    
    
    print("maxVal = ", maxVal)
    return maxVal

# exampleConsList = [(-68.39567*(1 - xp0)) >= (38-24.5)*(190 - zp0)]

def getDepthInterval(currImageSetConString, vert_x, vert_y, vert_z,currGroupRegionCons):
    env = gp.Env(empty=True)
    # env.setParam("WLSAccessID", str)
    # env.setParam("WLSSECRET", str)
    # env.setParam("LICENSEID", int)
    env.setParam("OutputFlag", 0)
    env.start()
    
    
    m = gp.Model( "model1", env=env)
    xp0 = m.addVar(lb=-GRB.INFINITY, name="xp0")
    yp0 = m.addVar(lb=-GRB.INFINITY, name="yp0")
    zp0 = m.addVar(lb=-GRB.INFINITY, name="zp0")
        # m.setParam(GRB.Param.NonConvex, 2)
    m.setParam(GRB.Param.NumericFocus, 3)
    # # # m.setParam(GRB.Param.OutputFlag, 0)
    m.setParam('BarHomogeneous',1)
    # m.setParam('Method',5)

    # Create variables
    x = m.addVar(lb=-GRB.INFINITY, name="x")
    y = m.addVar(lb=-GRB.INFINITY, name="y")
    z = m.addVar(lb=-GRB.INFINITY, name="z")

    # xp0 = m.addVar(lb=-GRB.INFINITY, name="xp0")
    # yp0 = m.addVar(lb=-GRB.INFINITY, name="yp0")
    # zp0 = m.addVar(lb=-GRB.INFINITY, name="zp0")

    minVal =0
    maxVal =0
    # Set objective: x
    obj = ((xp0 - x)**2 + (yp0 - y)**2 + (zp0 - z)**2)

    m.addConstr(x == vert_x, "c0")
    m.addConstr(y == vert_y, "c1")
    m.addConstr(z == vert_z, "c2")
    
    # print(vert_x, vert_y, vert_z)

    consList = currImageSetConString.split(",")
    # consList = currImageSetConString
    # print(consList)
    
    for i in range(0, len(consList)):
        consName = "q_"+str(i)
        # print(consName, " :", consList[i])
        currCons = consList[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            # print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        except:
            print("error occured")
            exit(0)
        # m.addConstr(currCons, "str(consName)")


    currRegionConsString = str(currGroupRegionCons)
    currRegionConsString = currRegionConsString.replace("And(","")
    currRegionConsString = currRegionConsString.replace(")","")
    currRegionConsString = currRegionConsString.replace("\n","")
    currRegionConsString = currRegionConsString.replace("  ","")
    print("currRegionConsString = ", currRegionConsString)
    
    regionConsList = currRegionConsString.split(",")
    print("regionConsList = ", regionConsList)
    print(regionConsList[0])
    for i in range(0, len(regionConsList)):
        consName = "qr_"+str(i)
        # print(consName, " :", consList[i])
        currCons = regionConsList[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            # print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        except:
            print("error occured")
            exit(0)
        # m.addConstr(currCons, "str(consName)")
        



    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    minVal = obj.getValue()

    # for v in m.getVars():
    #     print('%s %g' % (v.VarName, v.X))

    # print('Obj: %g' % obj.getValue())
    
    # print("\n\nminimum value got ", minVal, math.sqrt(minVal))

    # m.setObjective(obj, GRB.MAXIMIZE)
    # m.setParam('TimeLimit', 0.1*60)
    # m.setParam('BarHomogeneous',1)
    
    
    # m.optimize()
    # print(m.Status)
    # if (m.Status == 9):
    #     print("Gurobi timeout happened")
    #     maxVal = obj.getValue()
    #     print("maxVal = ",maxVal)
    #     sleep(2)

    # # for v in m.getVars():
    # #     print('%s %g' % (v.VarName, v.X))

    # # print('Obj: %g' % obj.getValue())
    # maxVal = obj.getValue()

    return minVal

# def getDepthInterval2(currImageSetConString, vert_x, vert_y, vert_z,currGroupRegionCons):
#     m2 = gp.Model("model2")

#     m2.setParam(GRB.Param.NonConvex, 2)
#     m2.setParam(GRB.Param.NumericFocus, 3)
#     # # m.setParam(GRB.Param.OutputFlag, 0)
#     # m.setParam('BarHomogeneous',1)
#     # m.setParam('Method',5)

#     # Create variables
#     x = m2.addVar(lb=-GRB.INFINITY, name="x")
#     y = m2.addVar(lb=-GRB.INFINITY, name="y")
#     z = m2.addVar(lb=-GRB.INFINITY, name="z")

#     xp0 = m2.addVar(lb=-GRB.INFINITY, name="xp0")
#     yp0 = m2.addVar(lb=-GRB.INFINITY, name="yp0")
#     zp0 = m2.addVar(lb=-GRB.INFINITY, name="zp0")

#     minVal =0
#     maxVal =0
#     # Set objective: x
#     obj = ((xp0 - x)**2 + (yp0 - y)**2 + (zp0 - z)**2)

#     m2.addConstr(x == vert_x, "c0")
#     m2.addConstr(y == vert_y, "c1")
#     m2.addConstr(z == vert_z, "c2")

#     consList = currImageSetConString.split(",")
#     # consList = currImageSetConString

#     for i in range(0, len(consList)):
#         consName = "q_"+str(i)
#         print(consName, " :", consList[i])
#         currCons = consList[i]
#         # exec(f"m.addConstr({currCons})")
#         try:
#             exec(f"m2.addConstr({currCons})")
        
#         except NotImplementedError:
#             # currCons = currCons.replace("<","+0.000000000000000001<=")
#             # currCons = currCons.replace(">","-0.0000000000000000001>=")
#             currCons = currCons.replace("<","<=")
#             currCons = currCons.replace(">",">=")
#             exec(f"m2.addConstr({currCons})")
#             print("Exception handled")
#             # return 0,0
#             # sleep(2)
#         except OverflowError:
            
#             print("overflow error")
            
#             sleep(20)
#             exit(0)
#         except:
#             print("error occured")
#             exit(0)
#         # m.addConstr(currCons, "str(consName)")
#         # m.addConstr(currCons, "str(consName)")
        
#     currRegionConsString = str(currGroupRegionCons)
#     currRegionConsString = currRegionConsString.replace("And(","")
#     currRegionConsString = currRegionConsString.replace(")","")
#     currRegionConsString = currRegionConsString.replace("\n","")
#     currRegionConsString = currRegionConsString.replace("  ","")
#     print("currRegionConsString = ", currRegionConsString)
    
#     regionConsList = currRegionConsString.split(",")
#     print("regionConsList = ", regionConsList)
#     print(regionConsList[0])
#     for i in range(0, len(regionConsList)):
#         consName = "qr_"+str(i)
#         # print(consName, " :", consList[i])
#         currCons = regionConsList[i]
#         # exec(f"m.addConstr({currCons})")
#         try:
#             exec(f"m.addConstr({currCons})")
        
#         except NotImplementedError:
#             # currCons = currCons.replace("<","+0.000000000000000001<=")
#             # currCons = currCons.replace(">","-0.000000000000000001>=")
#             currCons = currCons.replace("<","<=")
#             currCons = currCons.replace(">",">=")
#             exec(f"m.addConstr({currCons})")
#             # print("Exception handled")
#             # return 0,0
#             # sleep(2)
#         except OverflowError:
            
#             print("overflow error")
            
#             sleep(20)
#             exit(0)
#         except:
#             print("error occured")
#             exit(0)
#         # m.addConstr(currCons, "str(consName)")
        

#     # m.setObjective(obj, GRB.MINIMIZE)
#     # m.optimize()

#     # minVal = obj.getValue()

#     # # for v in m.getVars():
#     # #     print('%s %g' % (v.VarName, v.X))

#     # # print('Obj: %g' % obj.getValue())
    
#     # print("\n\nminimum value got ", minVal)

#     m2.setObjective(obj, GRB.MAXIMIZE)
#     m2.setParam('TimeLimit', 0.01*60)
#     # m.setParam('BarHomogeneous',1)
    
#     try:
#         m2.optimize()
#     except:
#         print("Tool error: Solve interrupted (error code 10005)")
#         maxVal = 1000000
#         # sleep(5)
#         return maxVal
        
        
        
#     print(m2.status)
#     if m2.status == 4 or m2.status == 9:
#         print("Ub solution")
#         maxVal = 1000000
#         # sleep(5)
#         return maxVal
#     maxVal = obj.getValue()
#     print("\n\maximum value got ", maxVal, math.sqrt(maxVal))
    
#     # print(m.Status)
#     if (m2.Status == 9):
#         print("Gurobi timeout happened")
#         maxVal = obj.getValue()
#         print("maxVal = ",maxVal)
#         print("Trying with minimization")
#         # sleep(3)
        
#         m3 = gp.Model("model3")

#         m3.setParam(GRB.Param.NonConvex, 2)
#         m3.setParam(GRB.Param.NumericFocus, 3)
#         # # m.setParam(GRB.Param.OutputFlag, 0)
#         # m.setParam('BarHomogeneous',1)
#         # m.setParam('Method',5)

#         # Create variables
#         x = m3.addVar(lb=-GRB.INFINITY, name="x")
#         y = m3.addVar(lb=-GRB.INFINITY, name="y")
#         z = m3.addVar(lb=-GRB.INFINITY, name="z")

#         xp0 = m3.addVar(lb=-GRB.INFINITY, name="xp0")
#         yp0 = m3.addVar(lb=-GRB.INFINITY, name="yp0")
#         zp0 = m3.addVar(lb=-GRB.INFINITY, name="zp0")

#         minVal =0
#         maxVal =0
#         # Set objective: x
#         obj = -((xp0 - x)**2 + (yp0 - y)**2 + (zp0 - z)**2)

#         m3.addConstr(x == vert_x, "c0")
#         m3.addConstr(y == vert_y, "c1")
#         m3.addConstr(z == vert_z, "c2")

#         consList = currImageSetConString.split(",")
#         # consList = currImageSetConString

#         for i in range(0, len(consList)):
#             consName = "q_"+str(i)
#             print(consName, " :", consList[i])
#             currCons = consList[i]
#             # exec(f"m.addConstr({currCons})")
#             try:
#                 exec(f"m3.addConstr({currCons})")
            
#             except NotImplementedError:
#                 # currCons = currCons.replace("<","+0.000000000000000001<=")
#                 # currCons = currCons.replace(">","-0.0000000000000000001>=")
#                 currCons = currCons.replace("<","<=")
#                 currCons = currCons.replace(">",">=")
#                 exec(f"m3.addConstr({currCons})")
#                 print("Exception handled")
#                 # return 0,0
#                 # sleep(2)
#             except OverflowError:
                
#                 print("overflow error")
                
#                 sleep(20)
#                 exit(0)
#             except:
#                 print("error occured")
#                 exit(0)
                
                
#         m3.setObjective(obj, GRB.MINIMIZE)
#         m3.setParam('TimeLimit', 0.01*60)
#         # m.setParam('BarHomogeneous',1)
        
        
#         m3.optimize()
#         maxVal = -obj.getValue()
#         print("\n\maximum value got ", maxVal, math.sqrt(maxVal))
        
#         if (m2.Status == 9):
#             print("Gurobi timeout happened again")
#             maxVal = 1000000
#             print("Timeout happened again returning 1000000 as maxval")
#             # sleep(2)
            
        
        
        
        
        
        
        
        
        
        

#     # for v in m.getVars():
#     #     print('%s %g' % (v.VarName, v.X))

#     # print('Obj: %g' % obj.getValue())
    

#     return maxVal


def getDepthInterval2(currImageSetConString, vert_x, vert_y, vert_z,currGroupRegionCons):
    env = gp.Env(empty=True)
    # env.setParam("WLSAccessID", str)
    # env.setParam("WLSSECRET", str)
    # env.setParam("LICENSEID", int)
    env.setParam("OutputFlag", 0)
    env.start()
    
    
    m = gp.Model( "model1", env=env)
    xp0 = m.addVar(lb=-GRB.INFINITY, name="xp0")
    yp0 = m.addVar(lb=-GRB.INFINITY, name="yp0")
    zp0 = m.addVar(lb=-GRB.INFINITY, name="zp0")
        # m.setParam(GRB.Param.NonConvex, 2)
    m.setParam(GRB.Param.NumericFocus, 3)
    # # # m.setParam(GRB.Param.OutputFlag, 0)
    m.setParam('BarHomogeneous',1)
    # m.setParam('Method',5)

    # Create variables
    x = m.addVar(lb=-GRB.INFINITY, name="x")
    y = m.addVar(lb=-GRB.INFINITY, name="y")
    z = m.addVar(lb=-GRB.INFINITY, name="z")

    # xp0 = m.addVar(lb=-GRB.INFINITY, name="xp0")
    # yp0 = m.addVar(lb=-GRB.INFINITY, name="yp0")
    # zp0 = m.addVar(lb=-GRB.INFINITY, name="zp0")

    minVal =0
    maxVal =0
    # Set objective: x
    obj = ((xp0 - x)**2 + (yp0 - y)**2 + (zp0 - z)**2)

    m.addConstr(x == vert_x, "c0")
    m.addConstr(y == vert_y, "c1")
    m.addConstr(z == vert_z, "c2")
    
    # print(vert_x, vert_y, vert_z)

    consList = currImageSetConString.split(",")
    # consList = currImageSetConString
    # print(consList)
    
    for i in range(0, len(consList)):
        consName = "q_"+str(i)
        # print(consName, " :", consList[i])
        currCons = consList[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            # print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        except:
            print("error occured")
            exit(0)
        # m.addConstr(currCons, "str(consName)")


    currRegionConsString = str(currGroupRegionCons)
    currRegionConsString = currRegionConsString.replace("And(","")
    currRegionConsString = currRegionConsString.replace(")","")
    currRegionConsString = currRegionConsString.replace("\n","")
    currRegionConsString = currRegionConsString.replace("  ","")
    print("currRegionConsString = ", currRegionConsString)
    
    regionConsList = currRegionConsString.split(",")
    print("regionConsList = ", regionConsList)
    print(regionConsList[0])
    for i in range(0, len(regionConsList)):
        consName = "qr_"+str(i)
        # print(consName, " :", consList[i])
        currCons = regionConsList[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            # print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        except:
            print("error occured")
            exit(0)
        # m.addConstr(currCons, "str(consName)")
        



    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

    minVal = obj.getValue()

    # for v in m.getVars():
    #     print('%s %g' % (v.VarName, v.X))

    # print('Obj: %g' % obj.getValue())
    
    # print("\n\nminimum value got ", minVal, math.sqrt(minVal))

    # m.setObjective(obj, GRB.MAXIMIZE)
    # m.setParam('TimeLimit', 0.1*60)
    # m.setParam('BarHomogeneous',1)
    
    
    # m.optimize()
    # print(m.Status)
    # if (m.Status == 9):
    #     print("Gurobi timeout happened")
    #     maxVal = obj.getValue()
    #     print("maxVal = ",maxVal)
    #     sleep(2)

    # # for v in m.getVars():
    # #     print('%s %g' % (v.VarName, v.X))

    # # print('Obj: %g' % obj.getValue())
    # maxVal = obj.getValue()

    return minVal

# tempString = "-5*yp0+23>=0, \
#             -5*zp0+973>=0, \
#             -67*xp0+20*zp0-3882>=0,\
#                 2*zp0-389>=0,\
#             2*yp0-9>=0, \
#             10*xp0-1>=0, \
#             446032970269918841442500154105960108452677559398580342531204223632812500000*xp0-111508242567479609337750771548209982597654743585735559463500976562500000000*yp0-156444400020046202685119627218842779103891160730199771933257579803466796875*zp0+30901097608214865008506245858035186465442304049579719035827940549281760411648>=0"


# minVal, maxVal = getDepthInterval(tempString, 1, 6, 190)

# print(minVal, maxVal)


# m.addConstr(18640183871009435028879002379633607576481235668097724555991590023040771484375*xp0-21303067227887086932031557266479455045593560669203725410625338554382324218750*yp0-13910584928943421230912293046211303249037882778793573379516601562500*zp0+95863802709139921281855567370715357663903522101174430064229129852916978417664>=0, "q6")


# # Add rotated cone: x^2 <= yz
# m.addConstr(x**2 <= y*z, "qc1")


# getDepthInterval(exampleConsList, 1, 4, 190)



# -4 1 0
# exampleConsList = " -100*xp0-39>=0,  -100*yp0+451>=0,  -250*zp0+48411>=0,\
#                    500*zp0-96817>=0,  2*yp0-9>=0, 5*xp0+2>=0 "

# getDepthInterval(exampleConsList, -4, 1, 0)


# exampleConsList = "  -100*xp0+11>=0,   -100*yp0+451>=0,   -100*zp0+19451>=0,   2*zp0-389>=0,   2*yp0-9>=0,   10*xp0-1>=0" 

# getDepthInterval2(exampleConsList, 4, 1, 200)


# exampleConsList = " -100*xp0-39>=0,  577987036974342865120617997729411271285193407803550933667796771153131025610341378113916260291578352005438697081007859424062189646065235137939453125*xp0-621863933153777689721624374111867485616548329096955502655253472596147252726426129312123412286503329802047138064047260286315577104687690734863281250*yp0-11559740739486857788735732765331449094269975073384061019920080859041031354039685722241223010763536821737113013952580331533681601285934448242187500*zp0+5245760228948520483681429126529969260613629890118366608181673981102538140153524258962385815547992649567216141747483560722466851394296311709041688576>=0, -125*zp0+23989>=0,500*zp0-95951>=0, 2*yp0-9>=0 "
# getDepthInterval2(exampleConsList, -8, 1, 0) 

















