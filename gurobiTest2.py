#!/usr/bin/env python3.7


import gurobipy as gp
from gurobipy import GRB
import math

from time import sleep
from decimal import Decimal

# Create a new model
# m = gp.Model("qcp")



# exampleConsList = [(-68.39567*(1 - xp0)) >= (38-24.5)*(190 - zp0)]

def getDepthInterval(currImageSetConString, vert_x, vert_y, vert_z):
    
    m = gp.Model("model1")
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
    
    print(vert_x, vert_y, vert_z)

    consList = currImageSetConString.split(",")
    # consList = currImageSetConString
    print(consList)
    
    for i in range(0, len(consList)):
        consName = "q_"+str(i)
        print(consName, " :", consList[i])
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
            print("Exception handled")
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
    
    print("\n\nminimum value got ",vert_x, vert_y, vert_z, minVal, math.sqrt(minVal))

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

def getDepthInterval2(currImageSetConString, vert_x, vert_y, vert_z):
    m2 = gp.Model("model2")

    m2.setParam(GRB.Param.NonConvex, 2)
    m2.setParam(GRB.Param.NumericFocus, 3)
    # # m.setParam(GRB.Param.OutputFlag, 0)
    # m.setParam('BarHomogeneous',1)
    # m.setParam('Method',5)

    # Create variables
    x = m2.addVar(lb=-GRB.INFINITY, name="x")
    y = m2.addVar(lb=-GRB.INFINITY, name="y")
    z = m2.addVar(lb=-GRB.INFINITY, name="z")

    xp0 = m2.addVar(lb=-GRB.INFINITY, name="xp0")
    yp0 = m2.addVar(lb=-GRB.INFINITY, name="yp0")
    zp0 = m2.addVar(lb=-GRB.INFINITY, name="zp0")

    minVal =0
    maxVal =0
    # Set objective: x
    obj = ((xp0 - x)**2 + (yp0 - y)**2 + (zp0 - z)**2)

    m2.addConstr(x == vert_x, "c0")
    m2.addConstr(y == vert_y, "c1")
    m2.addConstr(z == vert_z, "c2")

    consList = currImageSetConString.split(",")
    # consList = currImageSetConString

    for i in range(0, len(consList)):
        consName = "q_"+str(i)
        print(consName, " :", consList[i])
        currCons = consList[i]
        # exec(f"m.addConstr({currCons})")
        try:
            exec(f"m2.addConstr({currCons})")
        
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.000000000000000001<=")
            # currCons = currCons.replace(">","-0.0000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            exec(f"m2.addConstr({currCons})")
            print("Exception handled")
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
        # m.addConstr(currCons, "str(consName)")

    # m.setObjective(obj, GRB.MINIMIZE)
    # m.optimize()

    # minVal = obj.getValue()

    # # for v in m.getVars():
    # #     print('%s %g' % (v.VarName, v.X))

    # # print('Obj: %g' % obj.getValue())
    
    # print("\n\nminimum value got ", minVal)

    m2.setObjective(obj, GRB.MAXIMIZE)
    m2.setParam('TimeLimit', .01*60)
    m2.setParam('DualReductions',1)
    # m.setParam('BarHomogeneous',1)
    
    try:
        m2.optimize()
    except:
        print("Tool error: Solve interrupted (error code 10005)")
        maxVal = 1000000
        # sleep(5)
        return maxVal
        
        
        
    print(m2.status)
    if m2.status == 4:
        print("Ub solution")
        maxVal = 1000000
        sleep(5)
        return maxVal
    maxVal = obj.getValue()
    print("\n\maximum value got ",vert_x, vert_y, vert_z, maxVal, math.sqrt(maxVal))
    
    # print(m.Status)
    if (m2.Status == 9):
        print("Gurobi timeout happened")
        maxVal = obj.getValue()
        print("maxVal = ",maxVal)
      
        sleep(3)
      
        
        
        
        
        
        

    # for v in m.getVars():
    #     print('%s %g' % (v.VarName, v.X))

    # print('Obj: %g' % obj.getValue())
    

    return maxVal



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

# exampleConsList = " xp0 >= -1, \
#                     xp0 <= 1, \
#                     yp0 >= -1, \
#                     yp0 <= 1, \
#                     zp0 >= -1, \
#                     zp0 <= 1, \
#                     "

# exampleConsList = " xp0 >= -0.00000000001, xp0 <= 0.000000001, yp0 >= -23424234234234234233421, yp0 <= 2342342342342323423423S1, zp0 >= -0.4234242344242342432342323423424234234234234234, zp0 <= 0.2423412342342334234234234234234234234234232342341651"

# exampleConsList = "2960000000000003700000000000000*xp0-739999999999996300000000000000*yp0-589655802619554421094076104737*zp0+115734602497715319148615200640788>=0,-100*yp0+451>=0, 399600000000000000*xp0+99900000000000000*yp0+167240450807188561*zp0-33074385653365822010>=0,  500*zp0-96817>=0, 2*yp0-9>=0, -8880000000000011100000000000000*xp0-2220000000000000000000000000000*yp0-3716454462381971984871560564525*zp0+734986347852574653358929840593038>=0"

# getDepthInterval(exampleConsList, 1, 9, 190) 

# sleep(5)

#second level cube
# exampleConsList = "-100*xp0+61>=0, -100*yp0+451>=0, -250*zp0+48411>=0, 500*zp0-96817>=0, 2*yp0-9>=0, 5*xp0-3>=0"


# exampleConsList = "2960000000000003700000000000000*xp0-739999999999996300000000000000*yp0-589655802619554421094076104737*zp0+115734602497715319148615200640788>=0,-100*yp0+451>=0, 399600000000000000*xp0+99900000000000000*yp0+167240450807188561*zp0-33074385653365822010>=0,  500*zp0-96817>=0, 2*yp0-9>=0, -8880000000000011100000000000000*xp0-2220000000000000000000000000000*yp0-3716454462381971984871560564525*zp0+734986347852574653358929840593038>=0,             -zp0+1190>=0, -24975000000000000*yp0-8946268656716419*zp0+1924566044776119626>=0, -39960000000000049950000000000000*xp0+9989999999999950050000000000000*yp0+8544599451720975915077672860187*zp0-1673423895826985037164757843435631>=0, -399600000000000000*xp0+99900000000000000*yp0+85445994517209871*zp0-16734238958269871618>=0, 2960000000000003700000000000000*xp0-739999999999996300000000000000*yp0-589655802619554421094076104737*zp0+115734602497715319148615200640788>=0, zp0-191>=0, 11100000000000000*yp0+3976119402985075*zp0-766562686567164146>=0, 16650000000000000*xp0-4162500000000000*yp0-3316813889734999*zp0+651007139049649901>=0"

# exampleConsList = exampleConsList + ", zp0-191>=0, -11100000000000000*xp0-3976119402985075*zp0+788762686567164146>=0, -199800000000000000*xp0-49950000000000000*yp0-82889917758148043*zp0+16398434374048126012>=0, -zp0+1190>=0, 11100000000000000*xp0+3976119402985075*zp0-766562686567164146>=0, 399600000000000000*xp0+99900000000000000*yp0+167240450807188561*zp0-33074385653365822010>=0"


exampleConsList = "2960000000000003700000000000000*xp0-739999999999996300000000000000*yp0-589655802619554421094076104737*zp0+115734602497715319148615200640788>=0, -100*yp0+451>=0, 399600000000000000*xp0+99900000000000000*yp0+167240450807188561*zp0-33074385653365822010>=0, 500*zp0-96817>=0, 2*yp0-9>=0, -8880000000000011100000000000000*xp0-2220000000000000000000000000000*yp0-3716454462381971984871560564525*zp0+734986347852574653358929840593038>=0"

# getDepthInterval(exampleConsList, -1, 1, 190)
# getDepthInterval2(exampleConsList, -1, 1, 190) 
getDepthInterval(exampleConsList, 1, 9, 190)
getDepthInterval2(exampleConsList, 1, 9, 190) 