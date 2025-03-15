import gurobipy as gp
from gurobipy import GRB
import math
import sys

from time import sleep

def computeDepthsOfPixels(pixelToConsider, currTriangle,v0Vertex, v1Vertex, v2Vertex,
                                                           currAbsGroupRegionCons, currRegionMinMaxValues, 
                                                            currRegionCornerPoints,
                                                            regionConsListToGurobi, minMaxFlag):
    
    # print(v0Vertex, v1Vertex, v2Vertex)

    env = gp.Env(empty=True)
    # env.setParam("WLSAccessID", str)
    # env.setParam("WLSSECRET", str)
    # env.setParam("LICENSEID", int)
    env.setParam("OutputFlag", 0)
    env.setParam("DualReductions", 0)
    
    env.start()


    m = gp.Model("m", env=env)

    # m.setParam(GRB.Param.NonConvex, 2)
    # m.setParam(GRB.Param.NumericFocus, 3)
    # # # # m.setParam(GRB.Param.OutputFlag, 0)
    # m.setParam('BarHomogeneous',1)
    # m.setParam('Method',5)


    xp0 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="xp0")
    yp0 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="yp0")
    zp0 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="zp0")

    # Triangle vertices. (Here we declare them as variables and then constrain them to fixed values.)
    x0 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x0")
    y0 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y0")
    z0 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z0")
    x1 = m.addVar( lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x1")
    y1 = m.addVar( lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y1")
    z1 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z1")
    x2 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x2")
    y2 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y2")
    z2 = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z2")

    # The point on the triangle (expressed via barycentrics).
    x = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
    y = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")
    z = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z")

    a = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="a")
    b = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="b")
    c = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="c")

    m.addConstr(a+b+c == 1, "bary_triangle")

    m.addConstr(x0 == v0Vertex[0], "x0_val")
    m.addConstr(y0 == v0Vertex[1], "y0_val")
    m.addConstr(z0 == v0Vertex[2], "z0_val")
    m.addConstr(x1 == v1Vertex[0], "x1_val")
    m.addConstr(y1 == v1Vertex[1], "y1_val")
    m.addConstr(z1 == v1Vertex[2], "z1_val")
    m.addConstr(x2 == v2Vertex[0], "x2_val")
    m.addConstr(y2 == v2Vertex[1], "y2_val")
    m.addConstr(z2 == v2Vertex[2], "z2_val")

    for i in range(0, len(regionConsListToGurobi)):
        consName = "qr_"+str(i)
        # print(consName, " :", regionConsListToGurobi[i])
        currCons = regionConsListToGurobi[i]
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
    
    # # 4. Express a point on the triangle using barycentric coordinates.
    m.addConstr(x == a*x0 + b*x1 + c*x2, "bary_x")
    m.addConstr(y == a*y0 + b*y1 + c*y2, "bary_y")
    m.addConstr(z == a*z0 + b*z1 + c*z2, "bary_z")


    # wx = m.addVar(lb=-GRB.INFINITY, name="wx")  
    # wy = m.addVar(lb=-GRB.INFINITY, name="wy")

    # # Enforce the fraction constraint: w * (zl - zp0) = (xl - xp0)
    # m.addConstr(wx * (z - zp0) == (x - xp0), name="xfraction_constraint")
    # m.addConstr(wy * (z - zp0) == (y - yp0), name="yfraction_constraint")

    xpixel = pixelToConsider[0]
    ypixel = pixelToConsider[1]

    # pixelConsList = [ str(xpixel)+" <= -68.39567*wx+24.5", str(ypixel)+" <= 68.39567*wy+24.5",
    #                     str(xpixel+1)+" >= -68.39567*wx+24.5", str(ypixel+1)+" >= 68.39567*wy+24.5"
    #                 ]
    

    pixelConsList = [ str(xpixel-24.5)+"*(z-zp0) >= -68.39567*(x-xp0)",
                     "-68.39567*(x-xp0)"+ " >= "+str(xpixel+1-24.5)+"*(z-zp0)",
                    str(ypixel-24.5)+"*(z-zp0) >= 68.39567*(y-yp0)",
                    "68.39567*(y-yp0)"+ " >= "+str(ypixel+1-24.5)+"*(z-zp0)"
                    ]
    

    # # # pixelConsList.append("68.39567*(yl -yp0)  >= (49-24.5)*(zl -zp0) ")
    for i in range(0, len(pixelConsList)):
        consName = "qp_"+str(i)
        # print(consName, " :", pixelConsList[i])
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
            
           
            exit(0)
        except Exception as e:
            print(str(e))
            print("error occured")
            exit(0)

    # Set the objective: minimize squared Euclidean distance
    m.setObjective(( zp0 - z), GRB.MINIMIZE)

    # print("minmizing...")
    m.optimize()

    # print(m.Status)

    

    minVal = -1
    maxVal = 1000
    if m.status == GRB.OPTIMAL:
        opt_x = x.X
        opt_y = y.X
        opt_z = z.X
        min_sq_dist = m.objVal
        min_dist = (min_sq_dist)
        # print("Minimum distance from cube to point: {:.4f}".format(min_dist))
        # print("Achieved at point: ({:.4f}, {:.4f}, {:.4f})".format(opt_x, opt_y, opt_z))
        minVal =  min_dist
    # else:
    #     print(m.status)
    #     print("No optimal solution found for the minimum distance problem.")
    #     print("pixelToConsider : ", pixelToConsider)
        

    if minMaxFlag == 1:
        m.setObjective(( zp0 - z), GRB.MAXIMIZE)

        # print("Maximizing...")
        m.optimize()

        # print(m.Status)

    
        # print("current traingle: ", currTriangle)
        # print("current pixel: ", pixelToConsider)

        if m.status == GRB.OPTIMAL:
            opt_x = x.X
            opt_y = y.X
            opt_z = z.X
            maxDistance = m.objVal
            # min_dist = (min_sq_dist)
            # print("Maximum distance from cube to point: {:.4f}".format(maxDistance))
            # print("Achieved at point: ({:.4f}, {:.4f}, {:.4f})".format(opt_x, opt_y, opt_z))
            maxVal = maxDistance 
        # else:
        #     print("No optimal solution found for the maximum distance problem.")

    return [minVal, maxVal]




