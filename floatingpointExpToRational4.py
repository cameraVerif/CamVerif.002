from z3 import *
from fractions import Fraction
import random
from time import sleep
import numpy as np
import math


# def mygcd(a, b) :
     
#     if (a == 0) :
#         return b
         
#     return mygcd(b % a, a)

def lcm(a,b):
    # #print("computing lcm")
    # return a*b //mygcd(a,b)

    return np.lcm(a,b)

s = Solver()
set_option(rational_to_decimal=False)
set_option(precision=200)

xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
xp1,yp1,zp1 = Reals('xp1 yp1 zp1')

leftList =[]
rightList=[]

def findLCMandUpdateExp(exp,outList):
    #print("\n----------\n find and update expression by lcm")
    #print("expression received :", exp)
    
    numArgs = exp.num_args()
    # print ("num args: ", exp.num_args())
    
    if(numArgs >1):
        operator = exp.decl().name()
        left = exp.arg(0)
        right = exp.arg(1)
        if(operator == "-"):
            ##print("- oprator")
            ##print("current right = ",right)
            right = eval("-"+str(right))
            ##print("new right = ",right)
            
            
        #print("left, right: ", left, right)
        #print("operator: ",operator)
        #print("\n")
        
        if(left.num_args()==0 and right.num_args()==0):
            
            #print("\n----left.num_args()==0 and right.num_args()==0")
            #print("left, right: ",left,right)
            
           
            
            # if(type(left) == <class 'z3.z3.RatNumRef'>)
            #print( left.decl().name())
            #print( right.decl().name())
            
            currentList = []
            
            currNumerator =1
            currDenomenator = 1
            currVariable = "x"
            
            leftVariableflag =0
            rightVariableflag = 0
            
            if ( left.decl().name() == "xp1" or left.decl().name() == "yp1" or left.decl().name() == "zp1" ):
                #print("variable detected at left side :", left.decl().name())
                currVariable =left.decl().name()
                leftVariableflag =1
            else:
                #print("number detected1 at left side")
                if '.' in str(left):
                    #print("decimal point present")
                    sleep(10)
                    currNumber = float(str(left).replace("?","")).as_integer_ratio()
                    #print(currNumber)
                    currNumerator =currNumber[0]
                    currDenomenator = currNumber[1]
                else:
                    #print("no decimal point")
                    
                    
                    leftStr = str(left).split("/")
                    
                    ##print(leftStr[0])
                    
                    
                    currNumerator = int(int(str(leftStr[0]))//1)
                    if(len(leftStr) == 2):
                        ##print(leftStr[1])
                        currDenomenator = int(int(str(leftStr[1]))//1)
                    else:
                        ##print("no denomentor")
                        currDenomenator = 1
                    
                    
                
                #print(currNumerator)
                #print(currDenomenator)

            if ( right.decl().name() == "xp1" or right.decl().name() == "yp1" or right.decl().name() == "zp1" ):
                #print("variable detected at right side :", right.decl().name())
                currVariable =right.decl().name()
                rightVariableflag =1
            else:
                #print("number detected2 at right side")
                if '.' in str(right):
                    ##print("decimal2 point present")
                    sleep(10)
                    
                    currNumber = float(str(right).replace("?","")).as_integer_ratio()
                    ##print(currNumber)
                    currNumerator =currNumber[0]
                    currDenomenator = currNumber[1]
                else:
                    ##print("no decimal point")
                    currNumerator = (str(right).replace("?",""))
                    currDenomenator = 1

                    rightStr = str(right).split("/")
                    
                    ##print(rightStr[0])
                    
                    
                    currNumerator = int(int(str(rightStr[0]))//1)
                    
                    if(len(rightStr) == 2):
                        ##print(rightStr[1])
                        currDenomenator = int(int(str(rightStr[1]))//1)
                    else:
                        ##print("no denomentor")
                        currDenomenator = 1
                   
                
                
                ##print(currNumerator)
                ##print(currDenomenator)
            
            if(leftVariableflag == 0 or rightVariableflag == 0):
                currentList.append(currVariable)
                currentList.append(currNumerator)
                currentList.append(currDenomenator)
                outList.append(currentList)
            else:
                # eg: xp1+yp1
                ##print("both are variables")
                ##print(outList)
                currVariable =left.decl().name()
                ##print(currVariable)
                currentList.append(currVariable)
                currentList.append(currNumerator)
                currentList.append(currDenomenator)                
                outList.append(currentList)
                
                currVariable =right.decl().name()
                ##print(currVariable)
                currentList =[]
                currentList.append(currVariable)
                currentList.append(currNumerator)
                currentList.append(currDenomenator)
                outList.append(currentList)
                ##print(outList)
                
        else:
            findLCMandUpdateExp(left,outList)
            #print("outList = ", outList)
            findLCMandUpdateExp(right,outList)
            #print("outList = ", outList)
            # sleep(4)
            #print("both calls executed")
        
    
    elif(numArgs ==0):
        
        ##print("\n zero argument expression")
        ##print(exp)
        currentList = []
        currNumerator =1
        currDenomenator = 1
        currVariable = "x"
        
        if ( exp.decl().name() == "xp1" or exp.decl().name() == "yp1" or exp.decl().name() == "zp1" ):
                ##print("variable detected :", exp.decl().name())
                currVariable =exp.decl().name()
        else:
            ##print("Number detected3 :", exp.decl().name())
            if '.' in str(exp):
                    ##print("decimal point present")
                    # sleep(10)
                    currNumber = float(str(exp).replace("?","")).as_integer_ratio()
                    ##print(currNumber)
                    currNumerator =currNumber[0]
                    currDenomenator = currNumber[1]
            else:
                ##print("no decimal point")
                # currNumerator = (str(exp).replace("?",""))
                #     
                expStr = str(exp).split("/")
                currNumerator = int(int(str(expStr[0]))//1)
                    
                ##print(expStr[0])
                if(len(expStr) == 2):
                    ##print(expStr[1])
                    currDenomenator = int(int(str(expStr[1]))//1)
                else:
                    ##print("no denomentor")
                    currDenomenator = 1
                
               
                

                
                
                
            ##print(currNumerator)
            ##print(currDenomenator)
        
        currentList.append(currVariable)
        currentList.append(currNumerator)
        currentList.append(currDenomenator)
        ##print("current list : ",currentList)
        outList.append(currentList)
        
    elif(numArgs ==1):
        print("\n")
        # print("\n\n\n\n one argument expression, something wrong\n\n\n")
        # print(exp)
        
 

def floatingExpToRationalExp(exp):
    #print(exp)
    # print ("num args: ", exp.num_args())
    # print ("children: ", exp.children())
    # print ("operator: ", exp.decl())
    # print ("op name:  ", exp.decl().name())
    
    left = ""
    right = ""
    # left = exp.arg(0)
    # right = exp.arg(1)
    
    newOperator = exp.decl().name()
     
    if(exp.decl().name() == "not" or exp.decl().name() == "Not"):
        #print("not operator")
        #print("remove not operator")
        
        exp = exp.arg(0)
        ##print(exp)
        
        operator = exp.decl().name()
        left = exp.arg(0)
        right = exp.arg(1)
        #print(left, right)
        #print(operator)
        
       
        
        if(operator == "=="):
            newOperator = "!="
        elif(operator == "!="):
            newOperator = "=="
        elif(operator == "<"):
            newOperator = ">="
        elif(operator == "<="):
            newOperator = ">"
        elif(operator == ">"):
            newOperator = "<="
        elif(operator == ">="):
            newOperator = "<"
        
        #print("newOperator = ", newOperator)
        
        exp = eval("left "+newOperator+"right")
        #print(exp)
    # else:
        # print("'not' operator not present")
        
    
    ##print(exp)
    
    left = exp.arg(0)
    right = exp.arg(1)  
    newOperator = exp.decl().name()  
    #print("left, right :", left, right)
    
    
    
    
    leftLCM = 1
    rightLCM = 1
    
    ##print("Find LCM: left side of exp")
    global leftList 
    global rightList
    leftList =[]
    rightList=[]
    findLCMandUpdateExp(left,leftList)
    # sleep(4)
    ##print("\n\n===========processingright side of main expression")
    findLCMandUpdateExp(right,rightList)
    
    
    #print("leftList : ", leftList)
    #print("\n\n")
    #print("rightList : ", rightList)
    #print("\n\n")
    
    
    
    leftDenomList = []
    rightDenomList = []
    for m in range(0,len(leftList)):
        ##print(leftList[m][2])
        leftDenomList.append(leftList[m][2])
    
    for m in range(0,len(rightList)):
        ##print(rightList[m][2])
        rightDenomList.append(rightList[m][2])
    ##print(leftDenomList)
    ##print(rightDenomList)
    
    leftLCM = leftDenomList[0]
    rightLCM = rightDenomList[0]
    
    ##print("\n computing left lcm")
    ##print("current lcm : ", leftLCM)
    for l in range(1,len(leftDenomList)):
        ##print("currnet left denom value : ", leftDenomList[l])
        leftLCM = lcm(leftLCM, leftDenomList[l])
        ##print(leftLCM)
        
    ##print("\n computing right lcm")
    ##print("current lcm :",rightLCM)    
    for l in range(1,len(rightDenomList)):
        ##print("currnet right denom value : ", rightDenomList[l])
        rightLCM = lcm(rightLCM, rightDenomList[l])
        ##print(rightLCM)
    
    # leftLCM = np.lcm.reduce(leftDenomList)
    # rightLCM = np.lcm.reduce(rightDenomList)
    ##print("\nleftLCM = ", leftLCM)
    ##print("rightLCM = ", rightLCM)
        
    
    if(leftLCM == 1 and rightLCM ==1) :
        ##print("both has lcm ==1")
        ##print("\n final exp")
        ##print(exp)
        return exp
    
    newExp = ""
    
    if(leftLCM != 1 and rightLCM ==1) :
        ##print("left lcm !=1 and right lcm ==1")
        
        leftExp =""
        rightExp=""
        currExp = ""
        for m in range(0,len(leftList)):
            ##print(leftList[m])
            currVariable = leftList[m][0]
            currNumerator =  leftList[m][1]
            currDenomenator = leftList[m][2]
            if(currVariable == "x"):
                ##print("constant only no variable")
                # c0 = int(currLcm//(denomList[0]))
                ###_### currExp = int(currNumerator)*int(leftLCM//currDenomenator)
                currExp = str(currNumerator)+"*"+str(int(leftLCM//currDenomenator))
                ##print("currExp :",currExp)
            else:
                ##print("exp with variable")
                ###_###newNumerator = int(currNumerator)*int(leftLCM//currDenomenator)
                newNumerator = str(currNumerator)+"*"+str(int(leftLCM//currDenomenator))
                currExp = str(newNumerator)+"*"+str(currVariable)
                ##print("currExp :",currExp)
            if(leftExp == ""):
                leftExp = str(currExp)
            else:
                leftExp = leftExp+"+"+str(currExp)
            
        ##print("\nFinal leftexp : ",leftExp )
        
        
        for m in range(0,len(rightList)):
            ##print(rightList[m])
            currVariable = rightList[m][0]
            currNumerator =  rightList[m][1]
            currDenomenator = rightList[m][2]
            
            if(currVariable == "x"):
                ##print("constant only no variable")
                ###_###currExp = int(currNumerator)*int(leftLCM)
                currExp = str(currNumerator)+"*"+str(leftLCM)
                ##print("currExp :",currExp)
            else:
                ##print("exp with variable")
                ###_###newNumerator = int(currNumerator)*int(leftLCM)
                newNumerator = str(currNumerator)+"*"+str(leftLCM)
                currExp = str(newNumerator)+"*"+str(currVariable)
                ##print("currExp :",currExp)
            if(rightExp == ""):
                rightExp = str(currExp)
            else:
                rightExp = rightExp+"+"+str(currExp)
        

        ##print("\nFinal rightExp : ",rightExp )
        
        newExp = leftExp+newOperator+ rightExp
        ##print("newExp : ",newExp)
        
        exp = (newExp)    
        ##print("\n final exp")
        ##print(exp)
        return exp
    
    if(leftLCM == 1 and rightLCM !=1) :
        #print("left lcm ==1 and right lcm !=1")
        
        leftExp =""
        rightExp=""
        currExp = ""
        for m in range(0,len(rightList)):
            ##print(rightList[m])
            currVariable = rightList[m][0]
            currNumerator =  rightList[m][1]
            currDenomenator = rightList[m][2]
            if(currVariable == "x"):
                ##print("constant only no variable")
                currExp = str(currNumerator)+"*"+str(int(rightLCM//currDenomenator))
                ##print("currExp :",currExp)
            else:
                ##print("exp with variable")
                newNumerator = str(currNumerator)+"*"+str(int(rightLCM//currDenomenator))
                currExp = str(newNumerator)+"*"+str(currVariable)
                ##print("currExp :",currExp)
            if(rightExp == ""):
                rightExp = str(currExp)
            else:
                rightExp = rightExp+"+"+str(currExp)
            
        ##print("\nFinal rightExp : ",rightExp )
        
        
        for m in range(0,len(leftList)):
            ##print(leftList[m])
            currVariable = leftList[m][0]
            currNumerator =  leftList[m][1]
            currDenomenator = leftList[m][2]
            
            if(currVariable == "x"):
                ##print("constant only no variable")
                currExp = str(currNumerator)+"*"+str(rightLCM)
                ##print("currExp :",currExp)
            else:
                ##print("exp with variable")
                newNumerator = str(currNumerator)+"*"+str(rightLCM)
                currExp = str(newNumerator)+"*"+str(currVariable)
                ##print("currExp :",currExp)
            if(leftExp == ""):
                leftExp = str(currExp)
            else:
                leftExp = leftExp+"+"+str(currExp)
        

        ##print("\nFinal leftExp : ",leftExp )
        
        newExp = leftExp+newOperator+ rightExp
        ##print("newExp : ",newExp)
        
        exp = (newExp)    
        ##print("\n final exp")
        ##print(exp)
        return exp
    
    if(leftLCM != 1 and rightLCM !=1) :
        ##print("both lcm are not equal to 1")
        leftExp =""
        rightExp=""
        currExp = ""
        
        ##print("updating left coefficients")
        for m in range(0,len(leftList)):
            ##print(leftList[m])
            currNumerator =  leftList[m][1]
            currDenomenator = leftList[m][2]
            
            newNumerator = str(currNumerator)+"*"+str(int(leftLCM//currDenomenator))
            leftList[m][1] = newNumerator
            ##print("updated the numerator ")
            ##print(leftList[m])
            ##print("\n")
        
        ##print("updating right coefficients")
        for m in range(0,len(rightList)):
            ##print(rightList[m])
            currNumerator =  rightList[m][1]
            currDenomenator = rightList[m][2]
            
            newNumerator = str(currNumerator)+"*"+str(int(rightLCM//currDenomenator))
            rightList[m][1] = newNumerator
            ##print("updated the numerator ")
            ##print(rightList[m])
            ##print("\n")
         
        ##print("multiplying left coefficients with rightLCM")   
        for m in range(0,len(leftList)):
            ##print(leftList[m])
            currVariable = leftList[m][0]
            currNumerator =  leftList[m][1]
            currDenomenator = leftList[m][2]
            
            if(currVariable == "x"):
                ##print("constant only no variable")
                currExp = str(currNumerator)+"*"+str(rightLCM)
                ##print("currExp :",currExp)
            else:
                ##print("exp with variable")
                newNumerator = str(currNumerator)+"*"+str(rightLCM)
                currExp = str(newNumerator)+"*"+str(currVariable)
                ##print("currExp :",currExp)
            if(leftExp == ""):
                leftExp = str(currExp)
            else:
                leftExp = leftExp+"+"+str(currExp)
        ##print("\nFinal leftExp : ",leftExp )
        
        ##print("multiplying right coefficients with leftLCM")
        for m in range(0,len(rightList)):
            ##print(rightList[m])
            currVariable = rightList[m][0]
            currNumerator =  rightList[m][1]
            currDenomenator = rightList[m][2]
            
            if(currVariable == "x"):
                ##print("constant only no variable")
                currExp = str(currNumerator)+"*"+str(leftLCM)
                ##print("currExp :",currExp)
            else:
                ##print("exp with variable")
                newNumerator = str(currNumerator)+"*"+str(leftLCM)
                currExp = str(newNumerator)+"*"+str(currVariable)
                ##print("currExp :",currExp)
            if(rightExp == ""):
                rightExp = str(currExp)
            else:
                rightExp = rightExp+"+"+str(currExp)
        
        ##print("\nFinal rightExp : ",rightExp )
            
                    
        newExp = leftExp+newOperator+ rightExp
        ##print("newExp : ",newExp)     
        
        
        exp = newExp    
        ##print("\n final exp")
        ##print(exp)
        return exp




def converteToPPLExpression(exp):
    # set_option(rational_to_decimal=False)
    # set_option(precision=200)
    newExp = floatingExpToRationalExp(exp)
    # #print("input exp old exp ", exp)
    # #print("\n")
    # #print("return exp(newExp): ",newExp)
    # sleep(5)
    return newExp





# exp = 447128543283582814200000000000000000*xp1 +111782135820895653600000000000000000*yp1 + \
#         148486717732234618473451675753730597*zp1 >= \
#         14585838604106798022300872335568973485999/500

# converteToPPLExpression(exp)


# # exp = Not(1.5-20*xp1-3.5*yp1>= 25*xp1-40*yp1)
# exp = Not(1.5-20*xp1-3.5*yp1>= 2.25*xp1-4.25*yp1)
# exp = Not(100 > -10*xp1-20*yp1+30*zp1-40*xp1)
# # exp = Not(3124.5 <= -67*xp1 + 16*zp1+.5*xp1+.3*yp1)
# # exp = Not(3124.5 <= 16*xp1+yp1)

# # exp = -3885216848646058340961553880121832738161715469948512480681246032210140931178533357284417464956128455058590199780011018579927753238532069833212220109966411896709168702369870780216833003231421162879168215601464365157816019475128194321661142887855317885424642218478550461790880859395264090849764215818140655755996704101562500000000*xp1 +\
# #   -1456956318684474590665812903340788135304664205913982220126009493537306743272380631957366090265313650041942720285390175682248916541402799799144992628666510250556066505327617080502887403162990370425872057506255840308931960884267227271623734887780085276487666862349209157719361473618587682343772371496015693992376327514648437500000*yp1 +\
# #   -1267207744695630071699446953607607659690876678809139254863167606502913684655928238951294925985929099915999075277804924027057529815249028817668334212174521957080074422258522124610317948072398184656741746499438412811865762652444553781492674626921012800079521424333665817370998970224638124904004143900237977504730224609375*zp1 >=\
# #   -3399564988793006284517418484904011043255727372616379777788515582426684313598940393875435590922867697118636732032889115639866689043096094006327418178753659243277802830432941085056597221027796646451138872132212183517387319262478167493663573686095759183258927675161220426677339772779528272818453351456863847475279755974169417261857

# exp = xp1 >= -0.4

# exp = xp1 <= -0.39
# exp = yp1 >= 4.5
# exp =  yp1 <= 4.51
# exp = Not(zp1 <= 193.51)
# # exp = Not(zp1 >= 193.5)

# exp =  -67*xp1 + 15*zp1 >= 2942.5
# # exp = Not(1075 <= 67*yp1 + 4*zp1)
# # exp =  Not(-3353 <= 67*yp1 + -19*zp1)
# # exp =  67*yp1 + -18*zp1 >= -3173
# exp =  67*yp1 + 5*zp1 >= 1260


# # exp = 7/3*xp1-20/7*yp1+10/9*zp1+5/2*xp1 >= -0.4*yp1+17/7
# exp = 2/4*xp1+9/4*yp1+12/5*zp1<=10*xp1
# exp = Not(2/5*xp1+9*yp1+12*zp1<=10.5*xp1)


# newExp = converteToPPLExpression(exp)
# #print("input exp old exp ", exp)
# #print("return exp(newExp): ",newExp)
# exit(0)
# exp = xp1 <= -.89



# #print(exp)
# #print(result)

# # yp1 <= 451/100
# # zp1 <= 19351/100
# # zp1 >= 387/2
# # yp1 >= 9/2
# # xp1 >= -9/10
# # xp1 <= -89/100
# xp1 <= -89/100
