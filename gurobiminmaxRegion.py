import gurobipy as gp
from gurobipy import GRB
import math

from time import sleep
from decimal import Decimal, getcontext



def getSplitLength(RegionConsString, variable, numOfSplits):
    
    m = gp.Model("model1")
    # m.setParam(GRB.Param.NonConvex, 2)
    m.setParam(GRB.Param.NumericFocus, 3)
    # # # m.setParam(GRB.Param.OutputFlag, 0)
    # m.setParam('BarHomogeneous',1)
    # m.setParam('Method',5)
    
    xp0 = m.addVar(lb=-GRB.INFINITY, name="xp0")
    yp0 = m.addVar(lb=-GRB.INFINITY, name="yp0")
    zp0 = m.addVar(lb=-GRB.INFINITY, name="zp0")

    minVal =0
    maxVal =0
   
    consList = RegionConsString.split(",")
    
    for i in range(0, len(consList)):
        consName = "q_"+str(i)
        print(consName, " :", consList[i])
        currCons = consList[i]
        try:
            exec(f"m.addConstr({currCons})")
        except NotImplementedError:
            currCons = currCons.replace("<","+0.1<=")
            currCons = currCons.replace(">","-0.1>=")
            # currCons = currCons.replace("<","<=")
            # currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
    
    obj = xp0
    
    if variable == "yp0":
        obj = yp0
    elif variable == "zp0":
        obj = zp0
        
    
    
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.Status <= 2:    
        
        minVal =  minVal = obj.X
    else: 
        minVal = -2000
        print("Min value not found")

   
    
    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()


    if m.Status <= 2:            
        maxVal = obj.X
    else: 
        maxVal = 2000
        print("Max value not found")

    
    
    print("Variable = ", variable)
    print("\n\nminimum value got ", minVal)
    print("\n\maximum value got ", maxVal)
    
    getcontext().prec = 3
    res = Decimal(Decimal(maxVal)-Decimal(minVal))/int(numOfSplits)
    
    return float(minVal),float(maxVal), float(res)
    

# consString = "10*xp0>=1,100*xp0<=11,10*yp0>=45,100*yp0<=451, 10*zp0>=1945,100*zp0<=19451"

# res = getSplitLength(consString, "xp0",10)   

# print(res)

    
    
    
    
    
    