from calendar import c
from z3 import *
from pyparma import *
import os
from datetime import date, datetime
import sys
from time import sleep
from fractions import Fraction
import anytree

import pythonRenderAnImage2


from importlib import reload  # Python 3.4+
   
import createPoly 
import environment
import posInvariantRegion1

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import numpy as np
import cv2 
from tensorflow.keras.models import load_model

import floatingpointExpToRational4
import gurobiminmaxRegion
import singleTriangleInvRegions30
# import singleTriangleInvRegions9
import oldCodeForRefinement1

import json
from datetime import datetime

import numpy as np

import cv2
import onnx
import onnxruntime

from onnx import numpy_helper


vertices = environment.vertices
nvertices = environment.nvertices





# Define a function to find the ancestor of a node with more than one child
def find_ancestor_with_multiple_children(node):
    lengthOfthePath = 0
    while node.parent is not None:
        lengthOfthePath += 1
        parent = node.parent
        if len(parent.children) > 1:
            return parent, lengthOfthePath
        node = parent
    return None, lengthOfthePath


def getDNNOutput(inputImage):
    model = load_model('saved_models/3_2')
    
    dnnOutput = 1

    image = cv2.imread(inputImage)    
    image = cv2.resize(image, (49,49)).copy()

    a,b,c = image.shape
    image = image.reshape(1,a,b,c)
    image = image.astype(np.float32) / 255.0
    image2 = tf.convert_to_tensor(image)
    dnnOutput  = np.argmax(model.predict(image2))

    
    return dnnOutput 






##############onnx #############

# import onnx
# import onnxruntime
# import cv2 
# from tensorflow.keras.models import load_model
# import tensorflow as tf

# model_onnx = onnx.load('iisc_net1.onnx')
# onnx.checker.check_model(model_onnx)
def getDNNOutput_onnx(inputImage,networkName):
    dnnOutput = 1
    # model = onnx.load('iisc_net1.onnx')

    image = cv2.imread(inputImage)    
    image = cv2.resize(image, (49, 49)).copy()

    a, b, c = image.shape
    image = image.reshape(1, a, b, c)
    

    image = image.astype(np.float32) / 255.0
    # image2 = tf.convert_to_tensor(image)

    session = onnxruntime.InferenceSession(networkName)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # print(input_name)
    # print(output_name)

    result = session.run([output_name], {input_name: image})
    dnnOutput  = np.argmax(np.array(result).squeeze(), axis=0)
    
    return dnnOutput 


# getDNNOutput_onnx("images/image2.ppm", "OGmodel_pb_converted.onnx")





# print(onnx.helper.printable_graph(model.graph))

# getDNNOutput("images/minmax_0.ppm")

# getDNNOutput("images/collisionImage_00.ppm")




##########################onnxend #############







def checkForCollision(pathHullConString, currTriangle):
    print("checkForCollision==>    Checking collision with triangle "+str(currTriangle))
    s1 = Solver()
    set_param('parallel.enable', True)
    set_option(rational_to_decimal=True)
    set_option(precision=20)
    set_param('parallel.enable', True)
    s1.set("sat.local_search_threads", 26)
    s1.set("sat.threads", 26)
    s1.set("timeout",10)
    
    xp0, yp0, zp0 = Reals('xp0 yp0 zp0')
    
    s1.add(simplify(eval(pathHullConString)))
    
    xk, yk, zk = Reals('xk yk zk')
    u, v, w = Reals('u v w')
    
    s1.add(u+v+w == 1)
    s1.add(And(u>=0, v>=0, w>=0))
    
    x0 = vertices[nvertices[currTriangle*3+0]*3+0]  
    y0 = vertices[nvertices[currTriangle*3+0]*3+1] 
    z0 = vertices[nvertices[currTriangle*3+0]*3+2] 

    x1 = vertices[nvertices[currTriangle*3+1]*3+0] 
    y1 = vertices[nvertices[currTriangle*3+1]*3+1]
    z1 = vertices[nvertices[currTriangle*3+1]*3+2]

    x2 = vertices[nvertices[currTriangle*3+2]*3+0]
    y2 = vertices[nvertices[currTriangle*3+2]*3+1]
    z2 = vertices[nvertices[currTriangle*3+2]*3+2]


    s1.add(xk == (u*x0+v*x1+w*x2))
    s1.add(yk == (u*y0+v*y1+w*y2))
    s1.add(zk == (u*z0+v*z1+w*z2))
    
    s1.add(xp0 == xk)
    s1.add(yp0 == yk)
    s1.add(zp0 == zk)
    
    
    # while (True):
    solverResult = s1.check()
    if( solverResult == sat):
        del(s1)
        
        return 1
    elif( solverResult == unsat): 
        del(s1)
        return 0
    del(s1)
   
    checkForCollision(pathHullConString, currTriangle)


def  backTrackRegion( previousRegionName,previousDNNOuput, intersectionRegionConZ3):
   
    # currImageGroupCubeConString = environment.
    
    
    
    currGroup = previousRegionName
    
  
   
    
    
    singleImageCubeString = intersectionRegionConZ3
   
    
    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
    xp1,yp1,zp1 = Reals('xp1 yp1 zp1')
    
    
   
    
    
    
    
    newFormula1 = And(True)
    if(int(previousDNNOuput) == 0):
        
        newFormula1 = Exists([xp0,yp0,zp0],And( singleImageCubeString ,xp1==xp0+.5,yp1==yp0,zp1==zp0+0.866)) 
    elif(int(previousDNNOuput) == 1):
        
        newFormula1 = Exists([xp0,yp0,zp0],And( singleImageCubeString ,xp1==xp0,yp1==yp0,zp1==zp0+1)) 
    elif(int(previousDNNOuput) == 2):
        
        newFormula1 = Exists([xp0,yp0,zp0],And( singleImageCubeString ,xp1==xp0-.5,yp1==yp0,zp1==zp0+0.866)) 
    
    
    
    
    g  = Goal()
    set_option(rational_to_decimal=False)
    set_option(precision=400)
    g.add((newFormula1))
    
    t1 = Tactic('simplify')
    t2 = Tactic('qe')
    t  = Then(t2, t1)
   
   
    
    singleImageCubeStringInCurrRegion = str(t(g)[0])
    
    singleImageCubeStringInCurrRegion = singleImageCubeStringInCurrRegion.replace("xp1","xp0")
    singleImageCubeStringInCurrRegion = singleImageCubeStringInCurrRegion.replace("yp1","yp0")
    singleImageCubeStringInCurrRegion = singleImageCubeStringInCurrRegion.replace("zp1","zp0")
    singleImageCubeStringInCurrRegion = singleImageCubeStringInCurrRegion.replace("\n","")
    
    singleImageCubeStringInCurrRegion = singleImageCubeStringInCurrRegion.replace("[","")
    singleImageCubeStringInCurrRegion = singleImageCubeStringInCurrRegion.replace("]","")

    singleImageCubeStringInCurrRegion = "And("+str(singleImageCubeStringInCurrRegion)+")"
    singleImageCube = eval(singleImageCubeStringInCurrRegion)
   
    # global returnCurrImageConZ3
    # returnCurrImageConZ3 = singleImageCubeStringInCurrRegion
    
        
    currCubeCons = singleImageCube
    
    
    
    
    s1 = Solver()
    set_param('parallel.enable', True)
    set_option(rational_to_decimal=False)
    set_option(precision=20)
    set_param('parallel.enable', True)
    s1.set("sat.local_search_threads", 26)
    s1.set("sat.threads", 26)
    # s1.set("timeout", 100)
    
    s1.add(simplify(currCubeCons))
    

    
    i=-1
    while(s1.check() == sat):
        i +=1
        m = s1.model()
        posXp = (eval("m[xp0].numerator_as_long()/m[xp0].denominator_as_long()"))
        posYp = (eval("m[yp0].numerator_as_long()/m[yp0].denominator_as_long()"))
        posZp = (eval("m[zp0].numerator_as_long()/m[zp0].denominator_as_long()"))
        
       
        
        rendererPosFile = open("imagePoses.txt",'w')    
        rendererPosFile.write(str(1)+"\n");
        rendererPosFile.write("collisionImage2_"+str(i)+"\n"); 
        rendererPosFile.write(str(posXp)+"\n"\
            +str(posYp)+"\n"\
            +str(posZp)+",\n")
        
        rendererPosFile.close()
        
        print(str(datetime.now()))
        # # tempstring = "./renderImageAtPosClipped2"
        # tempstring = "./renderImageAtPosClipped6_image4"
        # os.system(tempstring)
        # # print("intervals Updated")
        # print(str(datetime.now()))
        
        pythonRenderAnImage2.renderAnImage(posXp,posYp, posZp,"collisionImage2_"+str(i)+"0")
    
        
        
        currImageDnnOutput= getDNNOutput("images/"+str("collisionImage2_"+str(i))+"0.ppm")
        # currImageDnnOutput = 1
       
        iisc_net_dnnoutput = getDNNOutput_onnx("images/"+str("collisionImage2_"+str(i))+"0.ppm",'OGmodel_pb_converted.onnx')
        
        
        
        currImageCons = And(True)
        if(int(iisc_net_dnnoutput) == int(previousDNNOuput)):
        #if(1==1):
            if(currGroup =="A" ):
                print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("backTrackAndCheck==>   True collision happended!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(str(currGroup))
                # print("imageName : ",imageName)
                # global numberOfCollisions
                # print("numberOfCollisions = ",numberOfCollisions)
                # numberOfCollisions += 1
                # global collisionFlag
                # collisionFlag = 1
                print(datetime.now())
                # print("time Taken = ", datetime.now() - programStartTime)
                print("program finished with collision")
                sleep(100)
                
                exit(0)
            
            else:
               
                
                #get current image region 
                currRegionPPL = environment.groupCube[currGroup] 
                # currRegionPPL = 
                currRegionPolyhedra = posInvariantRegion1.computePosInvariantRegion(posXp, posYp, posZp, m, currRegionPPL )
                currImageConsString = str(currRegionPolyhedra.minimized_constraints())
                
                
                
                currImageConsString = currImageConsString.replace("x0","xp0")
                currImageConsString = currImageConsString.replace("x1","yp0")
                currImageConsString = currImageConsString.replace("x2","zp0")
                currImageConsString = currImageConsString.replace(" = ","==")
                currImageConsString = currImageConsString.replace("Constraint_System {"," ")
                currImageConsString = currImageConsString.replace("}"," ")
                currImageConsString = "And("+currImageConsString+" )"
                currImageCons = eval(currImageConsString)
                
                
                intersectionRegionConZ3 = And(currImageCons,currCubeCons )
                print("currRegionCons = ", intersectionRegionConZ3)
                
                currImageRegionName = currGroup
                # print("\n\npreviousOfcurrImageGroup : ",previousOfcurrImageGroup)
                
                # print("currDnnOutput : ",currGroupDnnOutput)
                
                # # previousDNNOuput =  currGroup[currGroup.rfind("_")+1: ] 
                previousRegionName = currImageRegionName[0:currImageRegionName.rfind("_")]
                
                # print("previousGroup.rfind(\"\_\") : "+str(previousGroup.rfind("_")))
                previousDNNOuput =  currImageRegionName[currImageRegionName.rfind("_")+1: ]
                
                # 
                
               
                sleep(2)
                backTrackRegion( previousRegionName,previousDNNOuput, intersectionRegionConZ3)
        
        else:
            
            currRegionPPL = environment.groupCube[currGroup] 
            # currRegionPPL = 
            currRegionPolyhedra = posInvariantRegion1.computePosInvariantRegion(posXp, posYp, posZp, m, currRegionPPL )
            currImageConsString = str(currRegionPolyhedra.minimized_constraints())
            
            
            currImageConsString = currImageConsString.replace("x0","xp0")
            currImageConsString = currImageConsString.replace("x1","yp0")
            currImageConsString = currImageConsString.replace("x2","zp0")
            currImageConsString = currImageConsString.replace(" = ","==")
            currImageConsString = currImageConsString.replace("Constraint_System {"," ")
            currImageConsString = currImageConsString.replace("}"," ")
            currImageConsString = "And("+currImageConsString+" )"
            currImageCons = eval(currImageConsString)
           
                
            
        ##Add current image invariant region
        
        s1.add(Not(currImageCons))
        
        
    return
        
            
            
            
            
    


def checkForRandomPoints(currGroupName, triangle,currRegionPPL,currDnnOutput, numberOfRandomPointsToCheck = 5):
    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
    
    ConString = str(currRegionPPL)
    ConString = ConString.replace("x0","xp0")
    ConString = ConString.replace("x1","yp0")
    ConString = ConString.replace("x2","zp0")
    ConString = ConString.replace(" = ","==")
    ConString = ConString.replace("Constraint_System {"," ")
    ConString = ConString.replace("}"," ")
    ConString = "And("+str(ConString)+")"
    
    
    s = Solver()
    s.add(eval(ConString))
    
    for i in range(0, numberOfRandomPointsToCheck):
        global createPoly
        # createPoly = reload(createPoly) 
        if(s.check() == sat):
            m = s.model()
            posXp = (eval("m[xp0].numerator_as_long()/m[xp0].denominator_as_long()"))
            posYp = (eval("m[yp0].numerator_as_long()/m[yp0].denominator_as_long()"))
            posZp = (eval("m[zp0].numerator_as_long()/m[zp0].denominator_as_long()"))
            
            
            notTheCurrentPosCons1 = Or(xp0!= m[xp0], yp0!=m[yp0],zp0!= m[zp0])
            s.add(notTheCurrentPosCons1)
            
            
            rendererPosFile = open("imagePoses.txt",'w')    
            rendererPosFile.write(str(1)+"\n");
            rendererPosFile.write("collisionImage_"+str(i)+"\n"); 
            rendererPosFile.write(str(posXp)+"\n"\
                +str(posYp)+"\n"\
                +str(posZp)+",\n")
            
            rendererPosFile.close()
            
           
            # tempstring = "./renderImageAtPosClipped2"
            # tempstring = "./renderImageAtPosClipped6_image4"
            # os.system(tempstring)
            # print("intervals Updated")
            
            
            
            pythonRenderAnImage2.renderAnImage(posXp,posYp, posZp,"collisionImage_"+str(i)+"0")
    
            
          
       
            currImageDnnOutput= getDNNOutput("images/"+str("collisionImage_"+str(i))+"0.ppm")
           
            iisc_net_dnnoutput = getDNNOutput_onnx("images/"+str("collisionImage_"+str(i))+"0.ppm",'OGmodel_pb_converted.onnx')
        
            # print(currImageDnnOutput, previousDNNOuput)
            
          
        
            # print("currImageDnnOutput = ",currImageDnnOutput)
            # sleep(2)
            # currImageDnnOutput = 1
            currRegionPolyhedra = posInvariantRegion1.computePosInvariantRegion(posXp, posYp, posZp, m, currRegionPPL )
            currImageCons = currRegionPolyhedra.minimized_constraints()
            
            currImageSetConString = str(currRegionPolyhedra.minimized_constraints())
            currImageSetConString = currImageSetConString.replace("x0","xp0")
            currImageSetConString = currImageSetConString.replace("x1","yp0")
            currImageSetConString = currImageSetConString.replace("x2","zp0")
            currImageSetConString = currImageSetConString.replace(" = ","==")
            currImageSetConString = currImageSetConString.replace("Constraint_System {"," ")
            currImageSetConString = currImageSetConString.replace("}"," ")
            currImageSetConString = "And("+currImageSetConString+" )"
            currGroupCons = eval(currImageSetConString)
            
            currImageConsZ3 = currGroupCons

            s.add(Not(currImageConsZ3))
                
            if(iisc_net_dnnoutput == currDnnOutput):
            #if(1==1):
                # compute image invariant region and check for collision
                

                xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
                xp1,yp1,zp1 = Reals('xp1 yp1 zp1')
                if(currDnnOutput == 0):
                    
                    newFormula1 = Exists([xp0,yp0,zp0],And( currGroupCons ,xp1==xp0-.5,yp1==yp0,zp1==zp0-.866)) 
                elif(currDnnOutput == 1):
                    
                    newFormula1 = Exists([xp0,yp0,zp0],And( currGroupCons ,xp1==xp0,yp1==yp0,zp1==zp0-1)) 
                elif(currDnnOutput == 2):
                  
                    newFormula1 = Exists([xp0,yp0,zp0],And( currGroupCons ,xp1==xp0+.5,yp1==yp0,zp1==zp0-.866))  
                    
                
                
                
                set_option(rational_to_decimal=False)
                set_option(precision=10)
                g  = Goal()
                g.add((newFormula1))
                
                t1 = Tactic('simplify')
                t2 = Tactic('qe')
                t  = Then(t2, t1)
                
                oldExp = t(g)[0]
                # print(oldExp)
                updatedExpString =[]
                
                for n in range(0,len(t(g)[0])):
                    exp = str( t(g)[0][n])
                   
                    exp = exp.replace("xp1","xp0")
                    exp = exp.replace("yp1","yp0")
                    exp = exp.replace("zp1","zp0")
                    exp = exp.replace("\n", "")
                    
                    try:
                        updatedExpString.append(eval(exp))
                    except:
                       
                        exp = exp.replace("/","//")
                        updatedExpString.append(eval(exp))
                
                
                
                updateExp = []
                
               
                for n in range(0,len(t(g)[0])):
                    exp = t(g)[0][n]
                  
                    # exp = str(exp).replace("xp0","xp1")
                    # exp = str(exp).replace("yp0","yp1")
                    # exp = str(exp).replace("zp0","zp1")
                    # exp = str(exp).replace("\n", "")
                    # print(exp)
                    # print("\n\n")
                
                    try:
                        exp = eval(str(exp).replace("\n",""))
                    except:
                       
                        exit(0)
                    
                    newExp = floatingpointExpToRational4.converteToPPLExpression(exp)
                    newExp = str(newExp)
                    
                    newExp = newExp.replace("xp1","xp0")
                    newExp = newExp.replace("yp1","yp0")
                    newExp = newExp.replace("zp1","zp0")
                    newExp = newExp.replace("\n", "")
                  
                    updateExp.append(newExp)
            
                
                pd4 = NNC_Polyhedron(3)
                xp0 = Variable(0)
                yp0 = Variable(1)
                zp0 = Variable(2)
            
                
                conFile = open("createPoly.py","w")
                tempstring = "from pyparma import *\n\ndef getPoly():\n    xp0 = Variable(0)\n"
                tempstring += "    yp0 = Variable(1)\n"
                tempstring += "    zp0 = Variable(2)\n"
                tempstring += "    pd3 = NNC_Polyhedron(3)\n"
                for n in range(0,len(updateExp)):
                    tempstring += "    pd3.add_constraint("+str(updateExp[n]).replace("?","")+")\n"
                
                
                tempstring +="    return pd3\n"
                conFile.write(tempstring)
                
                conFile.close()
                
                createPoly = reload(createPoly)   
                pd4= createPoly.getPoly()

                
                
                pd5 = NNC_Polyhedron(3)
                pd5.add_constraints(currImageCons)
                
               
                pd5.poly_hull_assign(pd4)
                
              
                
                pathHullConString = pd5.minimized_constraints()  
                
                pathHullConString = str(pathHullConString)
                pathHullConString = pathHullConString.replace("x0","xp0")
                pathHullConString = pathHullConString.replace("x1","yp0")
                pathHullConString = pathHullConString.replace("x2","zp0")
                pathHullConString = pathHullConString.replace(" = ","==")
                pathHullConString = pathHullConString.replace("Constraint_System {"," ")
                pathHullConString = pathHullConString.replace("}"," ")
                pathHullConString = "And("+str(pathHullConString)+")"
                
               
            
                
                # print("\n\n")
                #global collisionFlag
                #collisionFlag = 0
                
                
                collision = checkForCollision(pathHullConString, triangle)
                # if collision == 1:
                if(collision == 1 and (currGroupName =="A_0" or currGroupName =="A_1" or currGroupName =="A_2" ) ):
                    print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    print("checkCollisionValidity==>True collision happended")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    print(str(currGroupName))
                    print("triangle id = "+str(triangle))
                    # global numberOfCollisions
                    # print("numberOfCollisions = ",numberOfCollisions)
                    # numberOfCollisions += 1
                    # print("\n\n\n")
                    # global collisionFlag
                    # collisionFlag = 1
                    print(datetime.now())
                    # print("time Taken = ", datetime.now() - programStartTime)
                    print("program finished with collision")
                    sleep(10)
                    
                    exit(0)
                    return 
                    
                elif(collision == 1 and (currGroupName !="A_0" or currGroupName !="A_1" or currGroupName !="A_2"  ) ):
                    # #              
                    
                    
                  
                    
                    pdOriginalImageRegion = NNC_Polyhedron(3)
                    pdOriginalImageRegion.add_constraints(currImageCons)
                    
                    
                    #triangle convexhull polyhedra
                    x0 = vertices[nvertices[triangle*3+0]*3+0]  
                    y0 = vertices[nvertices[triangle*3+0]*3+1] 
                    z0 = vertices[nvertices[triangle*3+0]*3+2] 

                    x1 = vertices[nvertices[triangle*3+1]*3+0] 
                    y1 = vertices[nvertices[triangle*3+1]*3+1]
                    z1 = vertices[nvertices[triangle*3+1]*3+2]

                    x2 = vertices[nvertices[triangle*3+2]*3+0]
                    y2 = vertices[nvertices[triangle*3+2]*3+1]
                    z2 = vertices[nvertices[triangle*3+2]*3+2]
                    
                    
                    
                    # if (isinstance(x, float)):
                    #     print("float x")
                    #     xf = str(Fraction(x).limit_denominator())
                    #     xl = xf.split('/')
                    #     px = int(xl[0])
                    #     # qx = int(xl[1])
                    #     if(len(xl) == 2):
                    #         qx = int(xl[1])
                    #     else:
                    #         qx = 1

                    # else:
                    #     px = x
                    #     qx = 1
                    # if (isinstance(y, float)):
                    #     print("float y")
                    #     yf = str(Fraction(y).limit_denominator())
                    #     yl = yf.split('/')
                    #     py = int(yl[0])
                    #     # qy = int(yl[1])
                    #     if(len(yl) == 2):
                    #         qy = int(yl[1])
                    #     else:
                    #         qy = 1
                    # else:
                    #     py = y
                    #     qy = 1
                    # if (isinstance(z, float)):
                    #     print("float z")
                    #     zf = str(Fraction(z).limit_denominator())
                    #     zl = zf.split('/')
                    #     pz = int(zl[0])
                    #     # qz = int(zl[1])
                    #     if(len(zl) == 2):
                    #         qz = int(zl[1])
                    #     else:
                    #         qz = 1
                    # else:
                    #     pz = z
                    #     qz = 1
                    
                    
                    
                    
                    x0 = int(x0*pow(10,7)) 
                    y0 = int(y0*pow(10,7)) 
                    z0 = int(z0*pow(10,7)) 
                    
                    x1 = int(x1*pow(10,7)) 
                    y1 = int(y1*pow(10,7)) 
                    z1 = int(z1*pow(10,7)) 

                    x2 = int(x2*pow(10,7)) 
                    y2 = int(y2*pow(10,7)) 
                    z2 = int(z2*pow(10,7)) 
                    
                    
                    
                    xp0 = Variable(0)
                    yp0 = Variable(1)
                    zp0 = Variable(2)
                    
                    trianglePolyhedron = NNC_Polyhedron(3,'empty')
                    trianglePolyhedron.add_generator(point( x0*xp0+y0*yp0+z0*zp0, pow(10,7) ))
                    trianglePolyhedron.add_generator(point( x1*xp0+y1*yp0+z1*zp0, pow(10,7) ))
                    trianglePolyhedron.add_generator(point( x2*xp0+y2*yp0+z2*zp0, pow(10,7) ))
                    
                    
                    # xp0 = Variable(0)
                    # yp0 = Variable(1)
                    # zp0 = Variable(2)
                    
                    # trianglePolyhedron = NNC_Polyhedron(3,'empty')
                    # trianglePolyhedron.add_generator(point( x0*xp0+y0*yp0+z0*zp0 ))
                    # trianglePolyhedron.add_generator(point( x1*xp0+y1*yp0+z1*zp0 ))
                    # trianglePolyhedron.add_generator(point( x2*xp0+y2*yp0+z2*zp0 ))
                   
                   
                 
                    
                    
                    # #find the exact intersecting region
                    pd5.intersection_assign(trianglePolyhedron)
                   
                    currIntersectionRegionConsString = str(pd5.minimized_constraints())
                    
                    currIntersectionRegionConsString = currIntersectionRegionConsString.replace("x0","xp0")
                    currIntersectionRegionConsString = currIntersectionRegionConsString.replace("x1","yp0")
                    currIntersectionRegionConsString = currIntersectionRegionConsString.replace("x2","zp0")
                    currIntersectionRegionConsString = currIntersectionRegionConsString.replace("Constraint_System {"," ")
                    currIntersectionRegionConsString = currIntersectionRegionConsString.replace("}"," ")

                    currIntersectionRegionConsList = currIntersectionRegionConsString.split(",")
                    
                    
                    
                    headerFilePre = "#include \"ppl.hh\" \nusing namespace Parma_Polyhedra_Library; \nusing namespace Parma_Polyhedra_Library::IO_Operators;\nusing namespace std;\nVariable xp0(0);\nVariable yp0(1);\nVariable zp0(2);\nNNC_Polyhedron grpPolyhedron(3);";

                    # #writes gropu frustum constraints 
                    pplGrpConsInputFile = open("pplTrianglePath.h",'w')
                    pplGrpConsInputFile.write(headerFilePre);
                    pplGrpConsInputFile.write("\nconst int numOfExpression ="+str(len(currIntersectionRegionConsList))+";\n\n");
                    
                    pplGrpConsInputFile.write("\nConstraint grpCon[numOfExpression] = {");  
                    for k in range(0, len(currIntersectionRegionConsList)):
                        pplGrpConsInputFile.write(str(currIntersectionRegionConsList[k])+",")
                    pplGrpConsInputFile.write("};\n\n"); 
                    
                    pplGrpConsInputFile.write("int dnnOutput ="+str(currDnnOutput) +";\n\n"); 
                    pplGrpConsInputFile.close()
                    
                    tempstring = "touch pplTrianglePath.cpp"
                 
                    os.system(tempstring)

                    tempstring = "gcc pplTrianglePath.cpp -o pplTrianglePath -L/home2/habeebp/opt/include/ -L/home2/habeebp/opt/lib/ -I/home2/habeebp/opt/include/ -lstdc++ -lppl -lgmpxx -lgmp"
                    
                    os.system(tempstring)

                    tempstring = "./pplTrianglePath"
                    os.system(tempstring)

                    pplOutputFilePtr = open("triangleHullRegionpolyhedron.txt",'r')
                   
                    preRegionpolyhedronConString = pplOutputFilePtr.read()
                    
                    pplOutputFilePtr.close()
                    
                    preRegionpolyhedronConString = str(preRegionpolyhedronConString)
                    preRegionpolyhedronConString = preRegionpolyhedronConString.replace("A","xp0")
                    preRegionpolyhedronConString = preRegionpolyhedronConString.replace("B","yp0")
                    preRegionpolyhedronConString = preRegionpolyhedronConString.replace("C","zp0")
                    preRegionpolyhedronConString = preRegionpolyhedronConString.replace(" = ","==")
                    preRegionpolyhedronConStringList = preRegionpolyhedronConString.split(",")
                    
                    
                 
                    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
                    newCons = And(True)
                    for r in range(0,len(preRegionpolyhedronConStringList)):
                        newCons = simplify(And(newCons,eval(str(preRegionpolyhedronConStringList[r]))))
                    
                    
                   
                    intersectionRegionConZ3 = And(currImageConsZ3,newCons)
                  
                    # print(simplify(intersectionRegionConZ3))
                    
                    currImageRegionName = currGroupName
                    # print("\n\npreviousOfcurrImageGroup : ",previousOfcurrImageGroup)
                   
                    # print("currDnnOutput : ",currGroupDnnOutput)
                    
                    # # previousDNNOuput =  currGroup[currGroup.rfind("_")+1: ] 
                    previousRegionName = currImageRegionName[0:currImageRegionName.rfind("_")]
                  
                    # print("previousGroup.rfind(\"\_\") : "+str(previousGroup.rfind("_")))
                    previousDNNOuput =  currImageRegionName[currImageRegionName.rfind("_")+1: ]
                    
                    # print("previousDNNOuput =  previousGroup[previousGroup.rfind(\"\_\")+1: ] : "+str(previousDNNOuput))
                    # # pplGrpConsInputFile.write("\nint previousDNNOutput  ="+str(previousDNNOuput)+";\n\n");
                            
                    # # ####17/03/2022### Identified a logical mistake##
                    # ##Need to intersect with the image region and back propagate the new region#####
                    # ##not the current region#####
                    
                    
                    
                    sleep(5)
                    
                    backTrackRegion( previousRegionName,previousDNNOuput, intersectionRegionConZ3)
                    
                
              
            else:
               
                sleep(10)

        else:
           
            break
    
   
    return
                    
                    
def renderImageAndGetDnnOutput(posXp,posYp, posZp, i):    
    rendererPosFile = open("imagePoses.txt",'w')    
    rendererPosFile.write(str(1)+"\n");
    rendererPosFile.write("collisionImage_"+str(i)+"\n"); 
    rendererPosFile.write(str(posXp)+"\n"\
        +str(posYp)+"\n"\
        +str(posZp)+",\n")
    
    rendererPosFile.close()
    
   
    # tempstring = "./renderImageAtPosClipped2"
    # tempstring = "./renderImageAtPosClipped6_image4"
    # os.system(tempstring)
    
    
    pythonRenderAnImage2.renderAnImage(posXp,posYp, posZp,"collisionImage_"+str(i)+"0")
   
    

    # currImageDnnOutput= getDNNOutput("images/"+str("collisionImage_"+str(i))+"0.ppm")
   
    iisc_net_dnnoutput = getDNNOutput_onnx("images/"+str("collisionImage_"+str(i))+"0.ppm",'OGmodel_pb_converted.onnx')

    # print(currImageDnnOutput, previousDNNOuput)
    
    return iisc_net_dnnoutput
                    
                
                
                
                
        
        
        
        
def splitRegionAndCheckCollisionValidity(currGroup, triangle,currRegionPPL,dnnOutput, variableToSplit, numberOfSplits):
    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
    
    ConString = str(currRegionPPL)
    ConString = ConString.replace("x0","xp0")
    ConString = ConString.replace("x1","yp0")
    ConString = ConString.replace("x2","zp0")
    ConString = ConString.replace(" = ","==")
    ConString = ConString.replace("Constraint_System {"," ")
    ConString = ConString.replace("}"," ")
    
    minVal,maxVal, intervalLength = gurobiminmaxRegion.getSplitLength(ConString, variableToSplit, numberOfSplits)
    
   
    
    #rearrange regions based on the priority
    
    
    originalRegionCons = ConString
    
    
    
    for i in range(1, numberOfSplits+1):
        currAbsGroupName = "split_"+str(environment.splitCount)
      
        currConString = originalRegionCons+", "+variableToSplit+">= "+str(minVal+(i-1)*intervalLength)+" ,"+ variableToSplit +"<="+str(minVal+ i*intervalLength)
        
        currCons = "And("+str(currConString)+")"
        
       
        
        xp0 = Variable(0)
        yp0 = Variable(1)
        zp0 = Variable(2)
        currSplitRegionPd = NNC_Polyhedron(3)
        currSplitRegionPd.add_constraints(currRegionPPL)
        
        tempMin = int(float(minVal+(i-1)*intervalLength)*pow(10,5)//1)
        tempMax = int(float(minVal+ i*intervalLength)*pow(10,5)//1)  
        
        
        if(variableToSplit == "xp0"):
            print("adding split region cons")
            currSplitRegionPd.add_constraint(pow(10,5)*xp0>= tempMin)
            currSplitRegionPd.add_constraint(pow(10,5)*xp0<= tempMax)
            
        
        environment.splitRegionPd[currAbsGroupName] = currSplitRegionPd.minimized_constraints()
       
        sleep(3)
        xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
        
        currRegionDnnoutputs = singleTriangleInvRegions9.computePixelIntervals(currAbsGroupName, eval(currCons), fromSplitRegion=1)

        environment.splitCount += 1
        # splitRegionAndCheckCollisionValidity(currGroup, triangle,currConString,dnnOutput, variableToSplit, 2)
        
     
        
        if dnnOutput in currRegionDnnoutputs:
            
            xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
            xp1,yp1,zp1 = Reals('xp1 yp1 zp1')
            currDnnOutput = dnnOutput
            currGroupCons = eval(currCons)
            currImageCons = environment.splitRegionPd[currAbsGroupName]
            if(currDnnOutput == 0):
                
                newFormula1 = Exists([xp0,yp0,zp0],And( currGroupCons ,xp1==xp0-.5,yp1==yp0,zp1==zp0-.866)) 
            elif(currDnnOutput == 1):
                
                newFormula1 = Exists([xp0,yp0,zp0],And( currGroupCons ,xp1==xp0,yp1==yp0,zp1==zp0-1)) 
            elif(currDnnOutput == 2):
                
                newFormula1 = Exists([xp0,yp0,zp0],And( currGroupCons ,xp1==xp0+.5,yp1==yp0,zp1==zp0-.866))  
                
            
          
            # sleep(2)
            
            set_option(rational_to_decimal=False)
            set_option(precision=10)
            g  = Goal()
            g.add((newFormula1))
            
            t1 = Tactic('simplify')
            t2 = Tactic('qe')
            t  = Then(t2, t1)
            
            oldExp = t(g)[0]
            # print(oldExp)
            updatedExpString =[]
            
            for n in range(0,len(t(g)[0])):
                exp = str( t(g)[0][n])
                print(exp)
                exp = exp.replace("xp1","xp0")
                exp = exp.replace("yp1","yp0")
                exp = exp.replace("zp1","zp0")
                exp = exp.replace("\n", "")
                
                try:
                    updatedExpString.append(eval(exp))
                except:
                   
                    exp = exp.replace("/","//")
                    updatedExpString.append(eval(exp))
            
            
            
            updateExp = []
            
           
            for n in range(0,len(t(g)[0])):
                exp = t(g)[0][n]
               
                
                # exp = str(exp).replace("xp0","xp1")
                # exp = str(exp).replace("yp0","yp1")
                # exp = str(exp).replace("zp0","zp1")
                # exp = str(exp).replace("\n", "")
                # print(exp)
                # print("\n\n")
            
                try:
                    exp = eval(str(exp).replace("\n",""))
                except:
                    
                    exit(0)
                
                newExp = floatingpointExpToRational4.converteToPPLExpression(exp)
                newExp = str(newExp)
                
                newExp = newExp.replace("xp1","xp0")
                newExp = newExp.replace("yp1","yp0")
                newExp = newExp.replace("zp1","zp0")
                newExp = newExp.replace("\n", "")
                
                updateExp.append(newExp)
            
            
            
            pd4 = NNC_Polyhedron(3)
            xp0 = Variable(0)
            yp0 = Variable(1)
            zp0 = Variable(2)
        
           
            conFile = open("createPoly.py","w")
            tempstring = "from pyparma import *\n\ndef getPoly():\n    xp0 = Variable(0)\n"
            tempstring += "    yp0 = Variable(1)\n"
            tempstring += "    zp0 = Variable(2)\n"
            tempstring += "    pd3 = NNC_Polyhedron(3)\n"
            for n in range(0,len(updateExp)):
                tempstring += "    pd3.add_constraint("+str(updateExp[n]).replace("?","")+")\n"
            
            
            tempstring +="    return pd3\n"
            conFile.write(tempstring)
            
            conFile.close()
            global createPoly
            createPoly = reload(createPoly)   
            pd4= createPoly.getPoly()

            
            
            pd5 = NNC_Polyhedron(3)
            pd5.add_constraints(currImageCons)
            
           
            
            pd5.poly_hull_assign(pd4)
            
           
            
            pathHullConString = pd5.minimized_constraints()  
            
            pathHullConString = str(pathHullConString)
            pathHullConString = pathHullConString.replace("x0","xp0")
            pathHullConString = pathHullConString.replace("x1","yp0")
            pathHullConString = pathHullConString.replace("x2","zp0")
            pathHullConString = pathHullConString.replace(" = ","==")
            pathHullConString = pathHullConString.replace("Constraint_System {"," ")
            pathHullConString = pathHullConString.replace("}"," ")
            pathHullConString = "And("+str(pathHullConString)+")"
            
          
            
            # print("\n\n")
            #global collisionFlag
            #collisionFlag = 0
            
           
            collision = checkForCollision(pathHullConString, triangle)
            
            
            if(collision == 1):
                print("Collision detected ")
                if(variableToSplit == "xp0"):
                    
                    splitRegionAndCheckCollisionValidity(currGroup, triangle,currSplitRegionPd.minimized_constraints(),dnnOutput, "yp0", numberOfSplits)
                elif(variableToSplit == "yp0"):
                    
                    splitRegionAndCheckCollisionValidity(currGroup, triangle,currSplitRegionPd.minimized_constraints(),dnnOutput, "zp0", numberOfSplits)
                else:
                   
                   
                    
                    checkForRandomPoints(currGroup, triangle,currSplitRegionPd.minimized_constraints(),dnnOutput, numberOfRandomPointsToCheck = 500)
                    
          
                    
                    
                    
                    
                    
        #     print("sleeping")
        #     sleep(100)
        
        
        
        # print("sleeping")
        # sleep(100)
        # break



def backTrackRandomPoints(xp, yp, zp, currGroup):
   
    currLen = len(currGroup)
    

    currGroupBackup = currGroup
    
    
    if(currLen != 1):
        
        dnnOutput = int(currGroup[str(currGroup).rfind("_")+1:])
        currGroup = currGroup[:str(currGroup).rfind("_")]
        
        # previousDnnOutput = int(currGroup[str(currGroup).rfind("_")+1:])
        
       
        # print("previousDnnOutput =", previousDnnOutput)
        
        if dnnOutput == 0:
            print(" old xp,yp,zp ", xp,yp,zp)
            xp = xp+0.5
            yp = yp
            zp = zp+0.866
           
        elif dnnOutput == 1:
            xp = xp
            yp = yp
            zp = zp+1
        else:
            xp = xp-0.5
            yp = yp
            zp = zp+0.866
        
        
        
        
        currentNodeString = str(currGroup)
        currentNode = anytree.find(environment.A, filter_=lambda node: node.name==currentNodeString)
        numOfCurrNodeChilds = len(currentNode.children)
        
        
        if numOfCurrNodeChilds > 1:
            currDnnOutput = renderImageAndGetDnnOutput(xp,yp,zp,currLen)      
            
            
            # print("previousDnnOutput = ", previousDnnOutput)
            
            if(int(currDnnOutput) == int(dnnOutput)) :
                      
                return backTrackRandomPoints(xp, yp, zp, currGroup)
            else:
                
                # sleep(7)
                return 0
        else:
            
            return backTrackRandomPoints(xp, yp, zp, currGroup)
            
    else:
        
        return 1
        # dnnOutput = currGroup[str(currGroup).rfind("_")+1:]
        # currDnnOutput = renderImageAndGetDnnOutput(xp,yp,zp,currLen)       
        # print("Current point Dnn Output = ", currDnnOutput)
        # print("dnnOutput to match = ", dnnOutput)
        # if(int(currDnnOutput) == int(dnnOutput)) :
        #     print("Dnn outputs match, back tracking again....")
        #     print("dnnOutput = ", dnnOutput)        
        #     return 1
        # else:
        #     print("Dnn outputs does not match, returnig to the collision prev region.")
        #     return 0
        

def randomPointsCheckLastRegion(intersectionRegionConZ3,dnnOutput, triangle, currGroup, numberOfRandomPointsToCheck=100 ):
    
    
    s = Solver()
    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
    s.add(intersectionRegionConZ3)  
   
    for i in range(0, numberOfRandomPointsToCheck):
        if(s.check() == sat):
            m= s.model()
            print(m)
            
            posXp = (eval("m[xp0].numerator_as_long()/m[xp0].denominator_as_long()"))
            posYp = (eval("m[yp0].numerator_as_long()/m[yp0].denominator_as_long()"))
            posZp = (eval("m[zp0].numerator_as_long()/m[zp0].denominator_as_long()"))
            
            
            notTheCurrentPosCons1 = Or(xp0!= m[xp0], yp0!=m[yp0],zp0!= m[zp0])
            s.add(notTheCurrentPosCons1)
            
            iisc_net_dnnoutput = renderImageAndGetDnnOutput(posXp,posYp, posZp, i)
            # print("OGmodel dnn output = ", currImageDnnOutput)
            
            
        
            if(iisc_net_dnnoutput == dnnOutput):
               
                backtrackStatus = backTrackRandomPoints(posXp, posYp, posZp, currGroup)
                if(backtrackStatus == 1):
                    
                    return 1
                
            
        else:
           
            return 2
   
    return 0


def backTrackARegionNStep(currentRegionCons,currentRegionPath, steps ):
    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
    xp1,yp1,zp1 = Reals('xp1 yp1 zp1')
    currRegionOutputToCheck = 3 #used to store the vlaue to check for interval image, in called function.

    for i in range(0, steps):
    # for i in range(0, 1):
      

        currBackTrackDnnOutput = int(currentRegionPath[currentRegionPath.rfind("_")+1:])
        currRegionOutputToCheck = currBackTrackDnnOutput
        
        #Propagate the current intersection region 1 step backward
        
        if(currBackTrackDnnOutput == 0):
           
            newFormula1 = Exists([xp0,yp0,zp0],And( currentRegionCons ,xp1==xp0+.5,yp1==yp0,zp1==zp0+.866)) 
        elif(currBackTrackDnnOutput == 1):
           
            newFormula1 = Exists([xp0,yp0,zp0],And( currentRegionCons ,xp1==xp0,yp1==yp0,zp1==zp0+1)) 
        elif(currBackTrackDnnOutput == 2):
           
            newFormula1 = Exists([xp0,yp0,zp0],And( currentRegionCons ,xp1==xp0-.5,yp1==yp0,zp1==zp0+.866))  
        
        


        
        set_option(rational_to_decimal=False)
        # set_option(precision=10)
        g  = Goal()
        g.add((newFormula1))
        
        t1 = Tactic('simplify')
        t2 = Tactic('qe')
        t  = Then(t2, t1)
             
        
        oldExp = t(g)[0]

        currentRegionCons = And(True)
        for n in range(0,len(t(g)[0])):
            newExp = str( t(g)[0][n])
            newExp = newExp.replace("xp1","xp0")
            newExp = newExp.replace("yp1","yp0")
            newExp = newExp.replace("zp1","zp0")
            newExp = newExp.replace("\n", "")
            currentRegionCons = And(currentRegionCons, eval(newExp))

        
        currentRegionPath = currentRegionPath[:currentRegionPath.rfind("_")]
        
    
    
    return currentRegionPath, currentRegionCons, currRegionOutputToCheck




       
def refineAndCheckCollisionValidity(intersectionRegionConZ3,dnnOutput, triangle, currGroup ):
    
    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
    xp1,yp1,zp1 = Reals('xp1 yp1 zp1')

    currentRegionCons = intersectionRegionConZ3
    currentRegionPath = currGroup
    
    
    
    status = randomPointsCheckLastRegion(currentRegionCons,dnnOutput, triangle, currentRegionPath )

    if status ==1:
        
        return 1
    elif(status == 2):
        
        return 0

   
    sleep(10)
    
    #Computing interval image of the current region
    #if the region's interval image fully classified to a non-dnnOutputToCheck value 
    # then we can declare this as a spurious collision
    #otherwise
    # if it is fully classified has the dnnoutputtocheck then we can backpropagate and check
    #else 
    # we need to split the region
   
    set_option(rational_to_decimal=False)
    
  
    
    z3Cons = ""
    
    s= Solver()
    s.add(simplify(currentRegionCons))
    
    #TODO: ensure the cons are conjuncts
    for c in s.assertions():
        print(" -- =>", str(c))
        
        conToAppend = str(c).replace("And","")
        conToAppend = conToAppend.replace("(","")
        conToAppend = conToAppend.replace(")","")
        conToAppend = conToAppend.replace("\n","")
        
        conToAppend = conToAppend.replace("xp0","xp1")
        conToAppend = conToAppend.replace("yp0","yp1")
        conToAppend = conToAppend.replace("zp0","zp1")
        
        
        z3Cons  += str(conToAppend)
    
    
    z3ConsList = z3Cons.split(",") 
    
    updateExp = []    
   
    for n in range(0,len(z3ConsList)):
        exp = z3ConsList[n]
           
       
        try:
            exp = eval(str(exp).replace("\n",""))
        except:
           
            exit(0)
        
        newExp = floatingpointExpToRational4.converteToPPLExpression(exp)
        newExp = str(newExp)
        
        newExp = newExp.replace("xp1","xp0")
        newExp = newExp.replace("yp1","yp0")
        newExp = newExp.replace("zp1","zp0")
        newExp = newExp.replace("\n", "")
        
        updateExp.append(newExp)
   
    
    pd4 = NNC_Polyhedron(3)
    xp0 = Variable(0)
    yp0 = Variable(1)
    zp0 = Variable(2)
   
    
    conFile = open("createPoly.py","w")
    tempstring = "from pyparma import *\n\ndef getPoly():\n    xp0 = Variable(0)\n\
    yp0 = Variable(1)\n\
    zp0 = Variable(2)\n\
    pd3 = NNC_Polyhedron(3)\n"
    for n in range(0,len(updateExp)):
        tempstring += "    pd3.add_constraint("+str(updateExp[n]).replace("?","")+")\n"
    
    
    tempstring +="    return pd3\n"
    conFile.write(tempstring)
    
    conFile.close()
    global createPoly
    createPoly = reload(createPoly)   
    pd4= createPoly.getPoly()
    
   
    currAbsGroupName = "split_"+str(environment.splitCount)
    environment.splitRegionPd[currAbsGroupName] = pd4.minimized_constraints()
   
    currentRegionPPLCons = pd4.minimized_constraints()
    sleep(3)
    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
    
    pd6 = NNC_Polyhedron(3)
    currBigRegionCons = environment.groupCube[currentRegionPath]    
    pd6.add_constraints(currBigRegionCons)

    pd6.intersection_assign(pd4)

    ConString = str(pd6.minimized_constraints())

    # ConString = str(pd4.minimized_constraints())
    ConString = ConString.replace("x0","xp0")
    ConString = ConString.replace("x1","yp0")
    ConString = ConString.replace("x2","zp0")
    ConString = ConString.replace(" = ","==")
    ConString = ConString.replace("Constraint_System {"," ")
    ConString = ConString.replace("}"," ")
    currCons = "And("+str(ConString)+")"
    
    
    currRegionDnnoutputs = singleTriangleInvRegions30.computePixelIntervals(currAbsGroupName, eval(currCons), fromSplitRegion=1)

    

    environment.splitCount += 1
    # splitRegionAndCheckCollisionValidity(currGroup, triangle,currConString,dnnOutput, variableToSplit, 2)
    
    if len(currRegionDnnoutputs) == 1 and int(dnnOutput) != int(currRegionDnnoutputs[0]):        
        return 0
   
    #######
    if environment.refineCount > 5:
        print("Going back to inv region appraoch")
        print("may be invalid collision")
        print(intersectionRegionConZ3,dnnOutput, triangle, currGroup)
        return 0

        foundARegion,foundConstraints = oldCodeForRefinement1.generateAndCheckOutputOfSmallRegions(currAbsGroupName, eval(currCons), int(dnnOutput), fromSplitRegion=1)

        if(foundARegion == 1):
            for currSmalReg in foundConstraints:
                currentNodeString = str(currGroup)
                currentNode = anytree.find(environment.A, filter_=lambda node: node.name==currentNodeString)
                ancNodeWithMultChild ="" 
                pathLengthToAncestor = 0
                # if(int(numOfCurrNodeChilds) == 1):
                ancNodeWithMultChild, pathLengthToAncestor  = find_ancestor_with_multiple_children(currentNode)
                if(ancNodeWithMultChild == None):
                    print("It is a valid collision,")
                    return 1            
                    
                
                # sleep(5)
                #bakcpropagate the region
                currentRegionPath, currentRegionCons, currRegionOutputToCheck = backTrackARegionNStep(And(currSmalReg,currentRegionCons),currGroup, pathLengthToAncestor )
            
                #refine the backtracked region
                smallCubeReturn = refineAndCheckCollisionValidity(And(currSmalReg,currentRegionCons),currRegionOutputToCheck, triangle, currentRegionPath)
                if smallCubeReturn == 1:            
                    return 1 
        else:
            print("may be invalid collision")
            return 0

        
    #############
    # sleep(5)
    #if region has a single dnn output and that is not as the currentDnnoutput of the path then
    #return spurious collision
    if len(currRegionDnnoutputs) == 1 and int(dnnOutput) != int(currRegionDnnoutputs[0]):
        
        return 0
    elif len(currRegionDnnoutputs) == 1 and int(dnnOutput) == int(currRegionDnnoutputs[0]):
        
        
        currentNodeString = str(currGroup)
        currentNode = anytree.find(environment.A, filter_=lambda node: node.name==currentNodeString)
        ancNodeWithMultChild ="" 
        pathLengthToAncestor = 0
        # if(int(numOfCurrNodeChilds) == 1):
        ancNodeWithMultChild, pathLengthToAncestor  = find_ancestor_with_multiple_children(currentNode)
        if(ancNodeWithMultChild == None):
            print("It is a valid collision,")
            return 1            
            
        
        #bakcpropagate the region
        print("backtrack the region")
        print(intersectionRegionConZ3,currGroup, pathLengthToAncestor)
        currentRegionPath, currentRegionCons, currRegionOutputToCheck = backTrackARegionNStep(intersectionRegionConZ3,currGroup, pathLengthToAncestor )
    
        #refine the backtracked region
        print("backtracking the region, because one output and is matching and mulitiple ancestoresssss")
        return refineAndCheckCollisionValidity(currentRegionCons,currRegionOutputToCheck, triangle, currentRegionPath)
    
        
        
    else:
       
        
        # ConString = str(currRegionPPL)
        # ConString = ConString.replace("x0","xp0")
        # ConString = ConString.replace("x1","yp0")
        # ConString = ConString.replace("x2","zp0")
        # ConString = ConString.replace(" = ","==")
        # ConString = ConString.replace("Constraint_System {"," ")
        # ConString = ConString.replace("}"," ")
        
        print("main splitting happening", environment.refineCount)  
        numberOfSplits = environment.numberOfSplit

    
        environment.refineCount += 1
        splitFlagXp = 1
        splitFlagYp = 1
        splitFlagZp = 1
        
        variableToSplit = "xp0"
        minXp,maxXp, xpIntervalLength = gurobiminmaxRegion.getSplitLength(ConString, variableToSplit, numberOfSplits)

        variableToSplit = "yp0"
        minYp,maxYp, ypIntervalLength = gurobiminmaxRegion.getSplitLength(ConString, variableToSplit, numberOfSplits)
       
        variableToSplit = "zp0"
        minZp,maxZp, zpIntervalLength = gurobiminmaxRegion.getSplitLength(ConString, variableToSplit, numberOfSplits)
       
        # if minXp == -2000 or maxXp == -2000:
        #     splitFlagXp = 0
        # if minYp == -2000 or maxYp == -2000:
        #     splitFlagYp = 0
        # if minZp == -2000 or maxZp == -2000:
        #     splitFlagZp = 0
        
        
        for i in range(1, environment.numberOfSplit+1):
            for j in range(1, environment.numberOfSplit+1):
                for k in range(1, environment.numberOfSplit+1):
                    #compute interval image of the small region
                    # tempMin = int(float(minVal+(i-1)*intervalLength)*pow(10,5)//1)
                    # tempMax = int(float(minVal+ i*intervalLength)*pow(10,5)//1)  
                   
                    
                    # currXp = int(float(minXp+(i-1)*xpIntervalLength)*pow(10,5)//1)
                    # currYp = int(float(minYp+(j-1)*ypIntervalLength)*pow(10,5)//1)
                    # currZp = int(float(minZp+(k-1)*zpIntervalLength)*pow(10,5)//1)
                    
                    currXp = minXp+(i-1)*xpIntervalLength
                    currYp = minYp+(j-1)*ypIntervalLength
                    currZp = minZp+(k-1)*zpIntervalLength
                    
                    print("splitting region",i,j,k)
                    #first check the small region is inside the currentRegion
                    s3= Solver()
                    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
                    s3.add(simplify(currentRegionCons))
                    
                    intervalCons = And(xp0>=currXp, xp0<=currXp+xpIntervalLength,
                                       yp0>=currYp, yp0<=currYp+ypIntervalLength,
                                       zp0>=currZp, zp0<=currZp+zpIntervalLength)
                    
                    s3.add(intervalCons)
                   
                    
                    
                    if(s3.check() == sat):
                        
                        sleep(5)
                        #create region polyhedra constraint    
                        currSmallCubeName = "split_"+str(environment.splitCount)    
                        xp0 = Variable(0)
                        yp0 = Variable(1)
                        zp0 = Variable(2)
                        currSplitRegionPd = NNC_Polyhedron(3)
                        currSplitRegionPd.add_constraints(currentRegionPPLCons)
                        
                        # currXp = int(float(minXp+(i-1)*xpIntervalLength)*pow(10,5)//1)
                        # currYp = int(float(minYp+(j-1)*ypIntervalLength)*pow(10,5)//1)
                        # currZp = int(float(minZp+(k-1)*zpIntervalLength)*pow(10,5)//1)
                        
                        tempxMin = int(float(currXp)*pow(10,5)//1)
                        tempxMax = int(float(currXp+xpIntervalLength)*pow(10,5)//1)  
                        
                        tempyMin = int(float(currYp)*pow(10,5)//1)
                        tempyMax = int(float(currYp+ypIntervalLength)*pow(10,5)//1) 
                        
                        tempzMin = int(float(currZp)*pow(10,5)//1)
                        tempzMax = int(float(currZp+zpIntervalLength)*pow(10,5)//1) 
                        
                       
                        currSplitRegionPd.add_constraint(pow(10,5)*xp0>= tempxMin)
                        currSplitRegionPd.add_constraint(pow(10,5)*xp0<= tempxMax)
                        
                        currSplitRegionPd.add_constraint(pow(10,5)*yp0>= tempyMin)
                        currSplitRegionPd.add_constraint(pow(10,5)*yp0<= tempyMax)
                        
                        currSplitRegionPd.add_constraint(pow(10,5)*zp0>= tempzMin)
                        currSplitRegionPd.add_constraint(pow(10,5)*zp0<= tempzMax)
                            
                        environment.splitRegionPd[currSmallCubeName] = currSplitRegionPd.minimized_constraints()
                        
                        # sleep(10)
                        xp0,yp0,zp0 = Reals('xp0 yp0 zp0')    
                        currSmallCubeDnnoutputs = singleTriangleInvRegions30.computePixelIntervals(currSmallCubeName, And(intervalCons,currentRegionCons), fromSplitRegion=1)
                        environment.splitCount += 1
                        
                        if len(currSmallCubeDnnoutputs) == 1 and int(dnnOutput) != int(currSmallCubeDnnoutputs[0]):
                           
                            # sleep(3)
                            continue
                        elif len(currSmallCubeDnnoutputs) == 1 and int(dnnOutput) == int(currSmallCubeDnnoutputs[0]):
                            
                            # sleep(100)
                            currentNodeString = str(currGroup)
                            currentNode = anytree.find(environment.A, filter_=lambda node: node.name==currentNodeString)
                            ancNodeWithMultChild ="" 
                            pathLengthToAncestor = 0
                            # if(int(numOfCurrNodeChilds) == 1):
                            ancNodeWithMultChild, pathLengthToAncestor  = find_ancestor_with_multiple_children(currentNode)
                            if(ancNodeWithMultChild == None):
                                print("It is a valid collision,")
                                return 1            
                                
                            
                            # sleep(5)
                            #bakcpropagate the region
                            currentRegionPath, currentRegionCons, currRegionOutputToCheck = backTrackARegionNStep(And(intervalCons,currentRegionCons),currGroup, pathLengthToAncestor )
                        
                            #refine the backtracked region
                            print("Backtracking from second step, because only one dnn output and multiple ancestors")
                            smallCubeReturn = refineAndCheckCollisionValidity(And(intervalCons,currentRegionCons),currRegionOutputToCheck, triangle, currentRegionPath)
                            if smallCubeReturn == 1:
                               
                                return 1 
                                      
                        else:
                            
                            sleep(10)
                            print("Refining the small cube againbecause more than one dnn output")
                            smallCubeReturn = refineAndCheckCollisionValidity(And(intervalCons,currentRegionCons),dnnOutput, triangle, currGroup)
                            if smallCubeReturn == 1:
                                
                                return 1 
                            
                    
    
        
    sleep(10)
    return 0                   
                        
                    
            
        
        
        
        
    
    
    
    





    

        








def checkValidityOfCollision(currGroup, triangle,currRegionPPL,dnnOutput,pd5):
   
    
    #first check if any of ancestor  of the current node has multiple childrens or current node has 
    #if not then the collision is valid
    #else back track the intersection region upto that level 
    #and then check for any of the random points inside that has a dnnoutput which is 
    #moving forward with the current path dnn output
    #then check any of the ancestors of the current node has multiple child
    #if not then return valid collision.
    #if some of the ancestor has multiple childs
    #   then back track the current point and check for initial region reachability
    #       if it can reach initial region then return valid collision
    #if random points check fails
    #   then partition the current intersection region to n parts
    #       then for each part
    #           if that part has multiple dnn outputs
    #               then partition again
    #                   continue with each part
    #           if the part has only single input 
    #               then if that input is current path input then back propagate that region upto 
    #               the ancestor who has multiple childs
    #               then check for the dnn outputs of the region if current path output only then back propagate next
    #                   ancestor with multiple child
    #               if the back propagation reaches at a point which has no multichild parent then valid collision
    #
    
 
    
    currentNodeString = str(currGroup)
    currentNode = anytree.find(environment.A, filter_=lambda node: node.name==currentNodeString)
    numOfCurrNodeChilds = len(currentNode.children)
   
    
    ancNodeWithMultChild ="" 
    pathLengthToAncestor = 0
    if(int(numOfCurrNodeChilds) == 1):
        ancNodeWithMultChild, pathLengthToAncestor  = find_ancestor_with_multiple_children(currentNode)
        if(ancNodeWithMultChild == None):
            
            return 1
        
        

    # exit()  
    environment.splitCount = 0
    #project the collide triangle and path intersection to the current region 
    #triangle convexhull polyhedra
    x0 = vertices[nvertices[triangle*3+0]*3+0]  
    y0 = vertices[nvertices[triangle*3+0]*3+1] 
    z0 = vertices[nvertices[triangle*3+0]*3+2] 

    x1 = vertices[nvertices[triangle*3+1]*3+0] 
    y1 = vertices[nvertices[triangle*3+1]*3+1]
    z1 = vertices[nvertices[triangle*3+1]*3+2]

    x2 = vertices[nvertices[triangle*3+2]*3+0]
    y2 = vertices[nvertices[triangle*3+2]*3+1]
    z2 = vertices[nvertices[triangle*3+2]*3+2]
    
    xp0 = Variable(0)
    yp0 = Variable(1)
    zp0 = Variable(2)


    x0 = int(x0*pow(10,7)) 
    y0 = int(y0*pow(10,7)) 
    z0 = int(z0*pow(10,7)) 
    
    x1 = int(x1*pow(10,7)) 
    y1 = int(y1*pow(10,7)) 
    z1 = int(z1*pow(10,7)) 

    x2 = int(x2*pow(10,7)) 
    y2 = int(y2*pow(10,7)) 
    z2 = int(z2*pow(10,7)) 
    
    # trianglePolyhedron = NNC_Polyhedron(3,'empty')
    # trianglePolyhedron.add_generator(point( x0*xp0+y0*yp0+z0*zp0 ))
    # trianglePolyhedron.add_generator(point( x1*xp0+y1*yp0+z1*zp0 ))
    # trianglePolyhedron.add_generator(point( x2*xp0+y2*yp0+z2*zp0 ))
    trianglePolyhedron = NNC_Polyhedron(3,'empty')
    trianglePolyhedron.add_generator(point( x0*xp0+y0*yp0+z0*zp0, pow(10,7) ))
    trianglePolyhedron.add_generator(point( x1*xp0+y1*yp0+z1*zp0, pow(10,7) ))
    trianglePolyhedron.add_generator(point( x2*xp0+y2*yp0+z2*zp0, pow(10,7) ))
    
    
   
   
    
    # #find the exact intersecting region
    pd5.intersection_assign(trianglePolyhedron)
    
    
    currIntersectionRegionConsString = str(pd5.minimized_constraints())
    
    currIntersectionRegionConsString = currIntersectionRegionConsString.replace("x0","xp0")
    currIntersectionRegionConsString = currIntersectionRegionConsString.replace("x1","yp0")
    currIntersectionRegionConsString = currIntersectionRegionConsString.replace("x2","zp0")
    currIntersectionRegionConsString = currIntersectionRegionConsString.replace("Constraint_System {"," ")
    currIntersectionRegionConsString = currIntersectionRegionConsString.replace("}"," ")

    currIntersectionRegionConsList = currIntersectionRegionConsString.split(",")
    
    
    
    headerFilePre = "#include \"ppl.hh\" \nusing namespace Parma_Polyhedra_Library; \nusing namespace Parma_Polyhedra_Library::IO_Operators;\nusing namespace std;\nVariable xp0(0);\nVariable yp0(1);\nVariable zp0(2);\nNNC_Polyhedron grpPolyhedron(3);";

    # #writes gropu frustum constraints 
    pplGrpConsInputFile = open("pplTrianglePath.h",'w')
    pplGrpConsInputFile.write(headerFilePre);
    pplGrpConsInputFile.write("\nconst int numOfExpression ="+str(len(currIntersectionRegionConsList))+";\n\n");
    
    pplGrpConsInputFile.write("\nConstraint grpCon[numOfExpression] = {");  
    for k in range(0, len(currIntersectionRegionConsList)):
        pplGrpConsInputFile.write(str(currIntersectionRegionConsList[k])+",")
    pplGrpConsInputFile.write("};\n\n"); 
    
    pplGrpConsInputFile.write("int dnnOutput ="+str(dnnOutput) +";\n\n"); 
    pplGrpConsInputFile.close()
    
    tempstring = "touch pplTrianglePath.cpp"
    
    os.system(tempstring)

    tempstring = "gcc pplTrianglePath.cpp -o pplTrianglePath -L/home2/habeebp/opt/include/ -L/home2/habeebp/opt/lib/ -I/home2/habeebp/opt/include/ -lstdc++ -lppl -lgmpxx -lgmp"
   
    os.system(tempstring)

    tempstring = "./pplTrianglePath"
    os.system(tempstring)

    pplOutputFilePtr = open("triangleHullRegionpolyhedron.txt",'r')
    
    preRegionpolyhedronConString = pplOutputFilePtr.read()
   
    pplOutputFilePtr.close()
    
    preRegionpolyhedronConString = str(preRegionpolyhedronConString)
    preRegionpolyhedronConString = preRegionpolyhedronConString.replace("A","xp0")
    preRegionpolyhedronConString = preRegionpolyhedronConString.replace("B","yp0")
    preRegionpolyhedronConString = preRegionpolyhedronConString.replace("C","zp0")
    preRegionpolyhedronConString = preRegionpolyhedronConString.replace(" = ","==")
    preRegionpolyhedronConStringList = preRegionpolyhedronConString.split(",")
    
    
    
    
    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
    newCons = And(True)
    for r in range(0,len(preRegionpolyhedronConStringList)):
        newCons = simplify(And(newCons,eval(str(preRegionpolyhedronConStringList[r]))))
    
    
   
    
    currImageSetConString = str(currRegionPPL)
    currImageSetConString = currImageSetConString.replace("x0","xp0")
    currImageSetConString = currImageSetConString.replace("x1","yp0")
    currImageSetConString = currImageSetConString.replace("x2","zp0")
    currImageSetConString = currImageSetConString.replace(" = ","==")
    currImageSetConString = currImageSetConString.replace("Constraint_System {"," ")
    currImageSetConString = currImageSetConString.replace("}"," ")
    currImageSetConString = "And("+currImageSetConString+" )"
    currGroupCons = eval(currImageSetConString)
   
    
    currImageConsZ3 = currGroupCons   
    
    
    intersectionRegionConZ3 = And(currImageConsZ3,newCons)
    
    
    
    
    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
    xp1,yp1,zp1 = Reals('xp1 yp1 zp1')
    currentRegionPath =  currGroup
    currentRegionCons = intersectionRegionConZ3
    currRegionOutputToCheck = dnnOutput

  
    environment.refineCount = 0
    
    if numOfCurrNodeChilds == 1:
        
        
        # sleep(2)
        currentRegionPath, currentRegionCons, currRegionOutputToCheck = backTrackARegionNStep(intersectionRegionConZ3,currGroup, pathLengthToAncestor )
    
      
        # dnnOutputToCheck = int(currentRegionPath[currentRegionPath.rfind("_")+1:])
       
        # 
    # sleep(10)
    
    
    # print("Going to split the region")
    sleep(10)

    refineStatus = refineAndCheckCollisionValidity(currentRegionCons,currRegionOutputToCheck, triangle, currentRegionPath)
    
    if refineStatus ==1:
        print("True collision detected !!!!!!!")
        return 1
    elif(refineStatus == 0):
        # print("Not a valid collision")
        return 0
    
    
    # sleep(100)
                    
    
    
 
    
        
    
    
    
    







