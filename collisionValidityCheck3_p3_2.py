from calendar import c
from z3 import *
from pyparma import *
import os
from datetime import date, datetime
import sys
from time import sleep
from fractions import Fraction
import anytree

import re
import pythonRenderAnImage2
import invariantRegionP3_1
import ast

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
# import singleTriangleInvRegions31_P3_1
# import singleTriangleInvRegions30_P3_cv_1
# import singleTriangleInvRegions9
import pyparmaFunctions1
import singleTriangleInvRegions30
# import intervalImageP3_4

import json
from datetime import datetime

import numpy as np

import cv2
import onnx
import onnxruntime

from onnx import numpy_helper

import invariantRegionP3_multiprocessing_3

from multiprocessing import Process, Queue
import signal
processes = []  # List to keep track of processes
numberOfProcessCreated = 0
uniqueIdList = []
num_processes =10
total_tasks = 100
isValidCollision = 0


smallCubeCount =0

vertices = environment.vertices
nvertices = environment.nvertices


def getMinMaxValuesOfList(inputList):

    min_first, max_first = float('inf'), float('-inf')
    min_second, max_second = float('inf'), float('-inf')
    min_third, max_third = float('inf'), float('-inf')

    for sublist in inputList:       

        first, second, third = sublist

        min_first = min(min_first, first)
        max_first = max(max_first, first)
        min_second = min(min_second, second)
        max_second = max(max_second, second)
        min_third = min(min_third, third)
        max_third = max(max_third, third)

    return [min_first, max_first, min_second, max_second, min_third, max_third]





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
    print("inputImage = ",inputImage)
    dnnOutput = 1

    image = cv2.imread(inputImage)    
    image = cv2.resize(image, (49,49)).copy()

    a,b,c = image.shape
    image = image.reshape(1,a,b,c)
    image = image.astype(np.float32) / 255.0
    image2 = tf.convert_to_tensor(image)
    dnnOutput  = np.argmax(model.predict(image2))

    print("dnnOutput = ", dnnOutput)
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
    # print(image.shape)

    image = image.astype(np.float32) / 255.0
    # image2 = tf.convert_to_tensor(image)

    session = onnxruntime.InferenceSession(networkName)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # print(input_name)
    # print(output_name)

    result = session.run([output_name], {input_name: image})
    dnnOutput  = np.argmax(np.array(result).squeeze(), axis=0)
    # print("networkName = ", networkName)
    # print("dnnOutput = ", dnnOutput)
    return dnnOutput 


# getDNNOutput_onnx("images/image2.ppm", "OGmodel_pb_converted.onnx")





# print(onnx.helper.printable_graph(model.graph))

# getDNNOutput("images/minmax_0.ppm")

# getDNNOutput("images/collisionImage_00.ppm")




##########################onnxend #############


def getHullCornerPoints(hullCons):
    gs = hullCons.minimized_generators()# // Use ph.minimized_generators() to minimal set of points for the polytope

    # print(gs)

    vertString = str(gs)
    vertString = vertString.replace("Generator_System {","").replace("}","").replace("point","").replace("(","").replace(")","")
    cornerPoints = vertString.split(",")
    # print(cornerPoints)  

    return cornerPoints




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
        # del(s1)
        print("collision detected. On the path ")
        return 1
    elif( solverResult == unsat): 
        # del(s1)
        return 0
    # del(s1)
    # print("timeout while checking collision of triangle"+str(currTriangle))
    # print("retrying... ")
    checkForCollision(pathHullConString, currTriangle)


def  backTrackRegion( previousRegionName,previousDNNOuput, intersectionRegionConZ3):
    # print("\n--//////////\nbackTrackAndCheck Header==>     back tracking reached previousGroup "+str(previousRegionName))
    # sleep(1)
    # print("prevCurrImageConZ3 = ",prevCurrImageConZ3)
    
    
    # currImageGroupCubeConString = environment.
    
    
    
    currGroup = previousRegionName
    
  
    # print("\n\nbackTrackAndCheck==>   singleImgeCube From ppl\n")
    
    
    singleImageCubeString = intersectionRegionConZ3
    print("moving single image frustum moving backward based on the previous dnn output")
    
    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
    xp1,yp1,zp1 = Reals('xp1 yp1 zp1')
    
    
    # print("\n singleImageCubeString : ",singleImageCubeString)
    
    
    
    # print("previous group dnn op = ", previousDNNOuput)
    newFormula1 = And(True)
    if(int(previousDNNOuput) == 0):
        # print("current group dnn op = ", previousDNNOuput)
        newFormula1 = Exists([xp0,yp0,zp0],And( singleImageCubeString ,xp1==xp0+.5,yp1==yp0,zp1==zp0+0.866)) 
    elif(int(previousDNNOuput) == 1):
        print("current group dnn op = ", previousDNNOuput)
        newFormula1 = Exists([xp0,yp0,zp0],And( singleImageCubeString ,xp1==xp0,yp1==yp0,zp1==zp0+1)) 
    elif(int(previousDNNOuput) == 2):
        print("current group dnn op = ", previousDNNOuput)
        newFormula1 = Exists([xp0,yp0,zp0],And( singleImageCubeString ,xp1==xp0-.5,yp1==yp0,zp1==zp0+0.866)) 
    
    # print("new formula ;")    
    # print(newFormula1)
    
    
    g  = Goal()
    set_option(rational_to_decimal=False)
    set_option(precision=400)
    g.add((newFormula1))
    
    t1 = Tactic('simplify')
    t2 = Tactic('qe')
    t  = Then(t2, t1)
    # print("\nt(g)[0]")
    # print (t(g)[0])
    
    
    print("\n\n single image cons in current previous region : t(g)[0][0]")
    print(t(g)[0][0])
    
    
    singleImageCubeStringInCurrRegion = str(t(g)[0])
    
    singleImageCubeStringInCurrRegion = singleImageCubeStringInCurrRegion.replace("xp1","xp0")
    singleImageCubeStringInCurrRegion = singleImageCubeStringInCurrRegion.replace("yp1","yp0")
    singleImageCubeStringInCurrRegion = singleImageCubeStringInCurrRegion.replace("zp1","zp0")
    singleImageCubeStringInCurrRegion = singleImageCubeStringInCurrRegion.replace("\n","")
    
    singleImageCubeStringInCurrRegion = singleImageCubeStringInCurrRegion.replace("[","")
    singleImageCubeStringInCurrRegion = singleImageCubeStringInCurrRegion.replace("]","")

    singleImageCubeStringInCurrRegion = "And("+str(singleImageCubeStringInCurrRegion)+")"
    singleImageCube = eval(singleImageCubeStringInCurrRegion)
    print("\n\n")
    print("After replacing constraints of image frustum from the next region in the currRegion")
    print(simplify(singleImageCube))
    
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
    
    print(s1.check())
    
    i=-1
    while(s1.check() == sat):
        i +=1
        m = s1.model()
        posXp = (eval("m[xp0].numerator_as_long()/m[xp0].denominator_as_long()"))
        posYp = (eval("m[yp0].numerator_as_long()/m[yp0].denominator_as_long()"))
        posZp = (eval("m[zp0].numerator_as_long()/m[zp0].denominator_as_long()"))
        
        print(m)
        
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
        
        iisc_net_dnnoutput = getDNNOutput_onnx("images/"+str("collisionImage2_"+str(i))+"0.ppm",environment.networkName)
        
     
        
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
                # sleep(100)
                
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
                
                currImageRegionName = currGroup
                # print("\n\npreviousOfcurrImageGroup : ",previousOfcurrImageGroup)
                # print("currDnnOutput : ",currGroupDnnOutput)
                
                # # previousDNNOuput =  currGroup[currGroup.rfind("_")+1: ] 
                previousRegionName = currImageRegionName[0:currImageRegionName.rfind("_")]
                # print("previousGroup.rfind(\"\_\") : "+str(previousGroup.rfind("_")))
                previousDNNOuput =  currImageRegionName[currImageRegionName.rfind("_")+1: ]
                
                # sleep(2)
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
            
          
            
            
            pythonRenderAnImage2.renderAnImage(posXp,posYp, posZp,"collisionImage_"+str(i)+"0")
    
            
          
            currImageDnnOutput= getDNNOutput("images/"+str("collisionImage_"+str(i))+"0.ppm")
            
            iisc_net_dnnoutput = getDNNOutput_onnx("images/"+str("collisionImage_"+str(i))+"0.ppm",environment.networkName)
        
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
                    # print(datetime.now())
                    # # print("time Taken = ", datetime.now() - programStartTime)
                    # print("program finished with collision")
                    # # sleep(10)
                    
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

                    # tempstring = "gcc pplTrianglePath.cpp -o pplTrianglePath -L/home2/habeebp/opt/include/ -L/home2/habeebp/opt/lib/ -I/home2/habeebp/opt/include/ -lstdc++ -lppl -lgmpxx -lgmp"
                    tempstring = "gcc pplTrianglePath.cpp -o pplTrianglePath -L/home/habeeb/projectFinal/softwares/ppl-1.2/include/ -L/home/habeeb/projectFinal/softwares/ppl-1.2/lib/ -I/home/habeeb/projectFinal/softwares/ppl-1.2/include/ -lstdc++ -lppl -lgmpxx -lgmp"
    
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
                    
                    
                    
                    # sleep(5)
                    
                    backTrackRegion( previousRegionName,previousDNNOuput, intersectionRegionConZ3)
                    
                
                else:
                    # print("checkCollisionValidity==> NO Collision with image"+str(i))
                    # print("\n\n\n")
                    pass
            else:
                
                pass
                # sleep(10)

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
    
   
    
    
    pythonRenderAnImage2.renderAnImage(posXp,posYp, posZp,"collisionImage_"+str(i)+"0")
    
   
    iisc_net_dnnoutput = getDNNOutput_onnx("images/"+str("collisionImage_"+str(i))+"0.ppm",environment.networkName)

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
            currSplitRegionPd.add_constraint(pow(10,5)*xp0>= tempMin)
            currSplitRegionPd.add_constraint(pow(10,5)*xp0<= tempMax)
        elif(variableToSplit == "yp0"):
            currSplitRegionPd.add_constraint(pow(10,5)*yp0>= tempMin)
            currSplitRegionPd.add_constraint(pow(10,5)*yp0<= tempMax)
        elif(variableToSplit == "zp0"):
            currSplitRegionPd.add_constraint(pow(10,5)*zp0>= tempMin)
            currSplitRegionPd.add_constraint(pow(10,5)*zp0<= tempMax)
            
        environment.splitRegionPd[currAbsGroupName] = currSplitRegionPd.minimized_constraints()
        
        xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
        # import singleTriangleInvRegions9


        cornerPointsOfProjection = pyparmaFunctions1.getPolyhedraCornerPoints(currSplitRegionPd.minimized_constraints())


        nextRegionMinMaxValues = getMinMaxValuesOfList(cornerPointsOfProjection)

        currRegionDnnoutputs = intervalImageP3_3.computeIntervalImage_P3(currAbsGroupName, eval(currCons),
                                            nextRegionMinMaxValues, cornerPointsOfProjection, fromSplitRegion=1)    



        # currRegionDnnoutputs = singleTriangleInvRegions9.computePixelIntervals(currAbsGroupName, eval(currCons), fromSplitRegion=1)

        # intervalImageP3_3.computeIntervalImage_P3(nextGroupName, tempFromula,
        #                                    nextRegionMinMaxValues, nextRegionCornerPoints)


        ###Update for P3#######




        environment.splitCount += 1
        # splitRegionAndCheckCollisionValidity(currGroup, triangle,currConString,dnnOutput, variableToSplit, 2)
        
        print("possible dnn outputs = ", currRegionDnnoutputs)
        
        if dnnOutput in currRegionDnnoutputs:
            print("same dnnoutput detected, checking for a possible collision")
            
            print("computing pathhull")
            xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
            xp1,yp1,zp1 = Reals('xp1 yp1 zp1')
            currDnnOutput = dnnOutput
            currGroupCons = eval(currCons)
            currImageCons = environment.splitRegionPd[currAbsGroupName]
            if(currDnnOutput == 0):
                print("currDnnOutput = ", currDnnOutput)
                newFormula1 = Exists([xp0,yp0,zp0],And( currGroupCons ,xp1==xp0-.5,yp1==yp0,zp1==zp0-.866)) 
            elif(currDnnOutput == 1):
                print("currDnnOutput = ", currDnnOutput)
                newFormula1 = Exists([xp0,yp0,zp0],And( currGroupCons ,xp1==xp0,yp1==yp0,zp1==zp0-1)) 
            elif(currDnnOutput == 2):
                print("currDnnOutput = ", currDnnOutput)
                newFormula1 = Exists([xp0,yp0,zp0],And( currGroupCons ,xp1==xp0+.5,yp1==yp0,zp1==zp0-.866))  
                
            
            # print("new formula --->")
            # print(newFormula1)
            # # sleep(2)
            
            set_option(rational_to_decimal=False)
            set_option(precision=10)
            g  = Goal()
            g.add((newFormula1))
            
            t1 = Tactic('simplify')
            t2 = Tactic('qe')
            t  = Then(t2, t1)
            print (t(g))
            
            # print("\n\n converting to PPL expression")
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
                    print("exception handled")
                    exp = exp.replace("/","//")
                    updatedExpString.append(eval(exp))
            
            
            
            updateExp = []
            
            # print("\n\n")
            for n in range(0,len(t(g)[0])):
                exp = t(g)[0][n]
                # print("current expression to conversion")
                # print(exp)
                
                # exp = str(exp).replace("xp0","xp1")
                # exp = str(exp).replace("yp0","yp1")
                # exp = str(exp).replace("zp0","zp1")
                # exp = str(exp).replace("\n", "")
                # print(exp)
                # print("\n\n")
            
                try:
                    exp = eval(str(exp).replace("\n",""))
                except:
                    # print("exception handled2 main_abs_1 @108")
                    exit(0)
                
                newExp = floatingpointExpToRational4.converteToPPLExpression(exp)
                newExp = str(newExp)
                
                newExp = newExp.replace("xp1","xp0")
                newExp = newExp.replace("yp1","yp0")
                newExp = newExp.replace("zp1","zp0")
                newExp = newExp.replace("\n", "")
                # print("\n\n")
                # print("returned expression ")
                # print(newExp)
                updateExp.append(newExp)
            # print("\n\n")
            # print("oldExp = ",oldExp)
            # print("updateExp = ",updateExp)
            
            
            pd4 = NNC_Polyhedron(3)
            xp0 = Variable(0)
            yp0 = Variable(1)
            zp0 = Variable(2)
        
            print("Opening create poly")
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

            # print("next region")
            # print(pd4.minimized_constraints())
            
            pd5 = NNC_Polyhedron(3)
            pd5.add_constraints(currImageCons)
            
            # print("current region cons :", pd5.minimized_constraints()) 
            # print("next region cons : ", pd4.minimized_constraints())
            
            pd5.poly_hull_assign(pd4)
            
            # print("path hull cons ", pd5.minimized_constraints())  
            
            pathHullConString = pd5.minimized_constraints()  
            
            pathHullConString = str(pathHullConString)
            pathHullConString = pathHullConString.replace("x0","xp0")
            pathHullConString = pathHullConString.replace("x1","yp0")
            pathHullConString = pathHullConString.replace("x2","zp0")
            pathHullConString = pathHullConString.replace(" = ","==")
            pathHullConString = pathHullConString.replace("Constraint_System {"," ")
            pathHullConString = pathHullConString.replace("}"," ")
            pathHullConString = "And("+str(pathHullConString)+")"
            
            # print("\n after replacing path hull cons\n")
            # print("\n\n",pathHullConString)
            
        
            
            # print("\n\n")
            #global collisionFlag
            #collisionFlag = 0
            
            # print("checking collision of pathHull with the triangle "+str(t))
            collision = checkForCollision(pathHullConString, triangle)
            
            # print("collision status = ", collision)
            
            if(collision == 1):
                # print("Collision detected ")
                if(variableToSplit == "xp0"):
                    # print("Current region is xp0")
                    # print("Splitting for yp0")
                    # sleep(3)
                    splitRegionAndCheckCollisionValidity(currGroup, triangle,currSplitRegionPd.minimized_constraints(),dnnOutput, "yp0", numberOfSplits)
                elif(variableToSplit == "yp0"):
                    # print("Current region is yp0")
                    # print("Splitting for zp0")
                    # sleep(3)
                    splitRegionAndCheckCollisionValidity(currGroup, triangle,currSplitRegionPd.minimized_constraints(),dnnOutput, "zp0", numberOfSplits)
                else:
                    ##TODO
                    # print("switch to old implementation and check.")
                   
                    # print("currGroup = ", currGroup)
                    # sleep(2)
                    checkForRandomPoints(currGroup, triangle,currSplitRegionPd.minimized_constraints(),dnnOutput, numberOfRandomPointsToCheck = 500)
                    # print("sleeping")
                    # sleep(3)
            else:
                # print("No collision with the current region")
                pass
                    
                    
                    
                    
                    
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
            # print(" old xp,yp,zp ", xp,yp,zp)
            xp = xp+0.5
            yp = yp
            zp = zp+0.866
            # print(" new xp,yp,zp ", xp,yp,zp)
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
            
            
            if(int(currDnnOutput) == int(dnnOutput)) :
                      
                return backTrackRandomPoints(xp, yp, zp, currGroup)
            else:
                
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
        

def randomPointsCheckLastRegion(intersectionRegionConZ3,dnnOutput, triangle, currGroup, numberOfRandomPointsToCheck=10 ):
    # print("Reached last region random points check")     
    
    s = Solver()
    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
    s.add(intersectionRegionConZ3)  

    currGroupCons = environment.groupCubeZ3[currGroup]
    s.add(currGroupCons)

    
    numberOfRandomPointsToCheck = 5
    for i in range(0, numberOfRandomPointsToCheck):
        if(s.check() == sat):
            m= s.model()
            # print(m)
            # sleep(3)
            
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
                    # print("Real collision detected!!!!!!!!!!!!!!!")
                    return 1
                else:
                    # print("This is not a valid collision, backtracking failed")
                    pass
                    
                # sleep(10)
            else:
                # print(f"Current random point's ({i}) dnn output does not match, continue with another point")
                # sleep(10)  
                pass   
            
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



  
            
        
        
       
def refineAndCheckCollisionValidity2(intersectionRegionConZ3,dnnOutput, triangle, currGroup, currDepth, fromBackTrack=0 ):
   
    global smallCubeCount
    global processes
    global numberOfProcessCreated
    global num_processes
    global uniqueIdList
    global isValidCollision
    
    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
    xp1,yp1,zp1 = Reals('xp1 yp1 zp1')

    currentRegionCons = intersectionRegionConZ3
    currentRegionPath = currGroup

  
    sCheck1= Solver()
    sCheck1.add(currentRegionCons)
    if(sCheck1.check() == unsat):
        # print("No valid region to refine")
        return 0 
    

    # sleep(5)
    status = randomPointsCheckLastRegion(currentRegionCons,dnnOutput, triangle, currentRegionPath )

    if status ==1:
        
        isValidCollision =1
        for process in processes:
            os.kill(process.pid, signal.SIGKILL) 
        return 1
    elif(status == 2):
       
        return 0

   
    if currDepth >=4:

        
        uniqueId = str(currGroup)+"_"+str(smallCubeCount)
        uniqueIdList.append(uniqueId)
        global num_processes
        # Ensure no more than `num_processes` processes are running
        while len(processes) >= 10:
            for proc in processes:
                if not proc.is_alive():  # Check if the process has finished
                    processes.remove(proc)

         
        process = Process(target=invariantRegionP3_multiprocessing_3.computeInvRegions, args=(currGroup, intersectionRegionConZ3, dnnOutput, uniqueId))
        processes.append(process)
        numberOfProcessCreated += 1
        process.start()

        return 4
        # sleep(5)
        valideRExist, invRList = invariantRegionP3_1.computeInvRegions(currGroup, intersectionRegionConZ3, dnnOutput, fromSplitRegion=0)

        if valideRExist == 1:
            print("Valid region found")

            for i in range(0, len(invRList)):
                print("invRList = ", invRList[i])
                currSetOfCons = invRList[i]
                print("Backpropagate the region into previous multichild parent")
        
                currentNodeString = str(currGroup)
                currentNode = anytree.find(environment.A, filter_=lambda node: node.name==currentNodeString)
                ancNodeWithMultChild ="" 
                pathLengthToAncestor = 0
                # if(int(numOfCurrNodeChilds) == 1):
                ancNodeWithMultChild, pathLengthToAncestor  = find_ancestor_with_multiple_children(currentNode)
                if(ancNodeWithMultChild == None):
                    print("It is a valid collision,")
                    return 1            
                    
                print("ancNodeWithMultChild = ", ancNodeWithMultChild)
                print("pathLengthToAncestor = ", pathLengthToAncestor)
                # sleep(5)
                #bakcpropagate the region
                currentRegionPath, currentRegionCons, currRegionOutputToCheck = backTrackARegionNStep(intersectionRegionConZ3,currGroup, pathLengthToAncestor )
            
                #refine the backtracked region
                return refineAndCheckCollisionValidity2(currentRegionCons,currRegionOutputToCheck, triangle, currentRegionPath, 0,1)



        else:
            print("No valid region found")
            return 0
   
        
    
    
    #Computing interval image of the current region
    #if the region's interval image fully classified to a non-dnnOutputToCheck value 
    # then we can declare this as a spurious collision
    #otherwise
    # if it is fully classified has the dnnoutputtocheck then we can backpropagate and check
    #else 
    # we need to split the region
    # print("Computing interval image of the current region.")
    set_option(rational_to_decimal=False)
    
    # print("Convert region to ppl polyhedra, for the interval image computation")
    # print("currentRegionCons = ", currentRegionCons)
    
    
    # z3Cons = ""
    
    # s= Solver()
    # s.add(simplify(currentRegionCons))
    # print("number of cons = ", len(s.assertions()))
    # #TODO: ensure the cons are conjuncts
    # for c in s.assertions():
    #     print(" -- =>", str(c))
        
    #     conToAppend = str(c).replace("And","")
    #     conToAppend = conToAppend.replace("(","")
    #     conToAppend = conToAppend.replace(")","")
    #     conToAppend = conToAppend.replace("\n","")
        
    #     conToAppend = conToAppend.replace("xp0","xp1")
    #     conToAppend = conToAppend.replace("yp0","yp1")
    #     conToAppend = conToAppend.replace("zp0","zp1")
        
        
    #     z3Cons  += str(conToAppend)
    
    
    # z3ConsList = z3Cons.split(",") 
    # print("z3ConsList = ", z3ConsList)
    # updateExp = []    

    z3Cons = ""
    z3ConsList = []
    s= Solver()
    s.add(simplify(currentRegionCons))
    
    #TODO: ensure the cons are conjuncts
    for c in s.assertions():
        
        # conToAppend = str(c).replace("And","")
        # conToAppend = conToAppend.replace("(","")
        # conToAppend = conToAppend.replace(")","")
        # conToAppend = conToAppend.replace("\n","")
        
        # conToAppend = conToAppend.replace("xp0","xp1")
        # conToAppend = conToAppend.replace("yp0","yp1")
        # conToAppend = conToAppend.replace("zp0","zp1")
        
        
        # z3Cons  += str(conToAppend)
    
        input_string = str(c)
        if input_string.startswith("And(") and input_string.endswith(")"):
            # print("The string starts with 'And(' and ends with ')'")
            trimmed_string = input_string[4:-1]  # Remove "And(" and the final ")"
            input_string = trimmed_string
            
        # Split the string into elements by commas, accounting for line breaks and indentation
        # elements = [element.strip() for element in trimmed_string.split(",") if element.strip()]
        # Split the string into elements by commas
        # elements = [element.strip().replace("\n", "").replace(" ", "") for element in trimmed_string.split(",") if element.strip()]

        elements = [
                element.strip()
                .replace("\n", "")
                .replace(" ", "")
                .replace("xp0", "xp1")
                .replace("yp0", "yp1")
                .replace("zp0", "zp1")
                for element in input_string.split(",")
                if element.strip()
            ]
        z3ConsList.extend(elements)

    # z3ConsList = z3Cons.split(",") 
    # print("z3ConsList = ", z3ConsList)
    updateExp = []    

   
    for n in range(0,len(z3ConsList)):
        exp = z3ConsList[n]
       
       
        try:
            exp = eval(str(exp).replace("\n",""))
        except:
            print("exception handled2 ")
            exit(0)
        
        newExp = floatingpointExpToRational4.converteToPPLExpression(exp)
        newExp = str(newExp)
        
        newExp = newExp.replace("xp1","xp0")
        newExp = newExp.replace("yp1","yp0")
        newExp = newExp.replace("zp1","zp0")
        newExp = newExp.replace("\n", "")
        newExp = newExp.replace(" ", "")
        
        # newExp =re.sub(r'(?<!<)(?<!>)=(?!=)', '==', newExp)
        newExp = re.sub(r'(?<![<>=!])=(?![<>=])', '==', newExp)
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
    
    
    currRegionCons = pd4.minimized_constraints()
    if(str(currRegionCons).replace(" ","") == "-1==0" or str(currRegionCons).replace(" ","") == "0==-1"):
        return 0
    
   
    currAbsGroupName = "split_"+str(environment.splitCount)
    environment.splitRegionPd[currAbsGroupName] = pd4.minimized_constraints()
   
    # print("\n\ngoing to compute interval image")
    currentRegionPPLCons = pd4.minimized_constraints()
    # sleep(3)
    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
    
    ConString = str(pd4.minimized_constraints())
    ConString = ConString.replace("x0","xp0")
    ConString = ConString.replace("x1","yp0")
    ConString = ConString.replace("x2","zp0")
    ConString = ConString.replace(" = ","==")
    ConString = ConString.replace("Constraint_System {"," ")
    ConString = ConString.replace("}"," ")
    currCons = "And("+str(ConString)+")"
   

    
    ###Update for P3#######
    # import singleTriangleInvRegions22
    # currRegionDnnoutputs = singleTriangleInvRegions30.computePixelIntervals(currAbsGroupName, eval(currCons), fromSplitRegion=1)
    
    cornerPointsOfProjection = pyparmaFunctions1.getPolyhedraCornerPoints(pd4.minimized_constraints())


    nextRegionMinMaxValues = getMinMaxValuesOfList(cornerPointsOfProjection)

   

    currRegionDnnoutputs = []
    # if currDepth <20 :
    #     currRegionDnnoutputs = intervalImageP3_4.computeIntervalImage_P3(currAbsGroupName, eval(currCons),
    #                                         nextRegionMinMaxValues, cornerPointsOfProjection, fromSplitRegion=1)
    
    # else:
    #     currRegionDnnoutputs = singleTriangleInvRegions30_P3_cv_1.computePixelIntervals(currAbsGroupName, eval(currCons), fromSplitRegion=1)
    
    if currDepth <= 3:
        # currRegionDnnoutputs = intervalImageP3_4.computeIntervalImage_P3(currAbsGroupName, eval(currCons),
        #                                     nextRegionMinMaxValues, cornerPointsOfProjection, fromSplitRegion=1)
        currRegionDnnoutputs = singleTriangleInvRegions30.computePixelIntervals(currAbsGroupName, eval(currCons), fromSplitRegion=1)
    # if currDepth == 9:  
    #     print("running p2") 
    #     print("currDepth = ", currDepth)     
    #     currRegionDnnoutputs = singleTriangleInvRegions30_P3_cv_1.computePixelIntervals(currAbsGroupName, eval(currCons), fromSplitRegion=1)
    #     print("using p2 method ",currRegionDnnoutputs)
    #     if(len(currRegionDnnoutputs) == 1):
    #         print("succeeded with p2")
    #     else:
    #         print("failed with p2")
    # intervalImageP3_3.computeIntervalImage_P3(currAbsGroupName, tempFromula,
    #                                        nextRegionMinMaxValues, cornerPointsOfProjection)

   

    # currRegionDnnoutputs = singleTriangleInvRegions31_P3_1.computeIntervalImage_P3(currAbsGroupName, eval(currCons), currRegionMinMaxValues, 
    #                         currRegionCornerPoints,fromSplitRegion=1):
    
    

    # sleep(2)
    
    #if region has a single dnn output and that is not as the currentDnnoutput of the path then
    #return spurious collision
    if len(currRegionDnnoutputs) == 1 and int(dnnOutput) != int(currRegionDnnoutputs[0]):
       
        # isValidCollision = 2
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
            isValidCollision = 1
            for process in processes:
                os.kill(process.pid, signal.SIGKILL) 

            return 1            
            
      
        # sleep(2)
        #bakcpropagate the region
        currentRegionPath, currentRegionCons, currRegionOutputToCheck = backTrackARegionNStep(intersectionRegionConZ3,currGroup, pathLengthToAncestor )
    
        #refine the backtracked region
        return refineAndCheckCollisionValidity2(currentRegionCons,currRegionOutputToCheck, triangle, currentRegionPath, 0,1)
                # refineAndCheckCollisionValidity2(eval(currConsToPass),dnnOutput, triangle, currGroup, currDepth+1,0)
    
        
        
    else:
        

        
        numberOfSplits = environment.numberOfSplit

        
        environment.totalNumRefinment += 1
        # sleep(5)
        
        variableToSplit = "xp0"
        # minXp,maxXp, xpIntervalLength = gurobiminmaxRegion.getSplitLength(ConString, variableToSplit, numberOfSplits)
        minXp,maxXp, xpIntervalLength = gurobiminmaxRegion.getSplitLength(ConString, variableToSplit, 2)
    
        variableToSplit = "yp0"
        minYp,maxYp, ypIntervalLength = gurobiminmaxRegion.getSplitLength(ConString, variableToSplit, 2)
        
        variableToSplit = "zp0"
        minZp,maxZp, zpIntervalLength = gurobiminmaxRegion.getSplitLength(ConString, variableToSplit, 1)
    
        
        # for i in range(1, environment.numberOfSplit+1):
        #     for j in range(1, environment.numberOfSplit+1):
        #         for k in range(1, environment.numberOfSplit+1):
        for i in range(1, 3):
            for j in range(1, 3):
                for k in range(1, 2):
                    #compute interval image of the small region
                    # tempMin = int(float(minVal+(i-1)*intervalLength)*pow(10,5)//1)
                    # tempMax = int(float(minVal+ i*intervalLength)*pow(10,5)//1)  
                    # print(f"\ni,j,k ==> {i} ,{j}, {k}, {currDepth}")
                    # currXp = int(float(minXp+(i-1)*xpIntervalLength)*pow(10,5)//1)
                    # currYp = int(float(minYp+(j-1)*ypIntervalLength)*pow(10,5)//1)
                    # currZp = int(float(minZp+(k-1)*zpIntervalLength)*pow(10,5)//1)
                    environment.refineCountNew += 1
                    
                    currXp = minXp+(i-1)*xpIntervalLength
                    currYp = minYp+(j-1)*ypIntervalLength
                    currZp = minZp+(k-1)*zpIntervalLength
                    
                    # print("minXp,maxXp, xpIntervalLength => ", minXp,maxXp, xpIntervalLength)
                    # print("minYp,maxYp, ypIntervalLength => ", minYp,maxYp, ypIntervalLength)
                    # print("minZp,maxZp, zpIntervalLength => ", minZp,maxZp, zpIntervalLength)
                    # print("currXp, currYp, currZp => ", currXp, currYp, currZp)
                    #first check the small region is inside the currentRegion
                    s3= Solver()
                    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
                    s3.add(simplify(eval(currCons)))
                    
                    intervalCons = And(xp0>=currXp, xp0<=currXp+xpIntervalLength,
                                       yp0>=currYp, yp0<=currYp+ypIntervalLength,
                                       zp0>=currZp, zp0<=currZp+zpIntervalLength)
                    
                    # print("intervalCons ==> ",intervalCons)
                    
                    s3.add(intervalCons)
                    # print("Smal cube region sat check")
                    
                    # print("intervalCons ==> ",intervalCons)
                    
                    # print("\nz3 assertions")
                    # for c in s3.assertions():
                    #     print(c)
                    # print("\n checking sat:")
                    # print(s3.check())
                    if(s3.check() == sat):
                        # print("There is atleast one point is inside the intersecting region")
                        # sleep(5)
                        
                        #create region polyhedra constraint    
                        currSmallCubeName = "split_"+str(environment.splitCount)    
                        environment.splitCount += 1
                        
                        xp0 = Variable(0)
                        yp0 = Variable(1)
                        zp0 = Variable(2)
                        currSplitRegionPd = NNC_Polyhedron(3)
                        currSplitRegionPd.add_constraints(currentRegionPPLCons)
                        

                        # currXp = int(float(minXp+(i-1)*xpIntervalLength)*pow(10,5)//1)
                        # currYp = int(float(minYp+(j-1)*ypIntervalLength)*pow(10,5)//1)
                        # currZp = int(float(minZp+(k-1)*zpIntervalLength)*pow(10,5)//1)

                        #workied
                        # tempxMax = int(float(currXp+xpIntervalLength)*pow(10,17)//1)  
                        
                        tempxMin = int(float(currXp)*pow(10,17)//1)
                        tempxMax = int(float(currXp+xpIntervalLength)*pow(10,17)//1)  
                        
                        tempyMin = int(float(currYp)*pow(10,17)//1)
                        tempyMax = int(float(currYp+ypIntervalLength)*pow(10,17)//1) 
                        
                        tempzMin = int(float(currZp)*pow(10,17)//1)
                        tempzMax = int(float(currZp+zpIntervalLength)*pow(10,17)//1) 
                        
                        # print("tempxMin, tempxMax ",tempxMin, tempxMax)
                        # print(tempyMin, tempyMax)
                        # print(tempzMin, tempzMax)
                        # print(currSplitRegionPd.minimized_constraints())
                       
                        # print("adding split region cons")
                        currSplitRegionPd.add_constraint(pow(10,17)*xp0>= tempxMin)
                        currSplitRegionPd.add_constraint(pow(10,17)*xp0<= tempxMax)
                        
                        currSplitRegionPd.add_constraint(pow(10,17)*yp0>= tempyMin)
                        currSplitRegionPd.add_constraint(pow(10,17)*yp0<= tempyMax)
                        
                        currSplitRegionPd.add_constraint(pow(10,17)*zp0>= tempzMin)
                        currSplitRegionPd.add_constraint(pow(10,17)*zp0<= tempzMax)
                            
                        environment.splitRegionPd[currSmallCubeName] = currSplitRegionPd.minimized_constraints()
                        # print("Polyhedra constrain after adding small cube cons = ")
                        # print(currSplitRegionPd.minimized_constraints())
                        # # sleep(2)
                        xp0,yp0,zp0 = Reals('xp0 yp0 zp0')    

                        ###Update for P3#######
                        # cornerPointsOfProjection = pyparmaFunctions1.getPolyhedraCornerPoints(currSplitRegionPd.minimized_constraints())


                        # nextRegionMinMaxValues = getMinMaxValuesOfList(cornerPointsOfProjection)

                        # print("cornerPointsOfProjection = ", cornerPointsOfProjection)
                        # print("nextRegionMinMaxValues = ", nextRegionMinMaxValues)
                        # print("going to call the refineandcheck function")
                      

                        constraintToPassString = str(currSplitRegionPd.minimized_constraints())
                        constraintToPassString = constraintToPassString.replace("x0","xp0")
                        constraintToPassString = constraintToPassString.replace("x1","yp0")
                        constraintToPassString = constraintToPassString.replace("x2","zp0")
                        constraintToPassString = constraintToPassString.replace(" = ","==")
                        constraintToPassString = constraintToPassString.replace("Constraint_System {"," ")
                        constraintToPassString = constraintToPassString.replace("}"," ")
                        currConsToPass = "And("+str(constraintToPassString)+")"
                        
                        # sleep(2)

                        smallCubeReturn = refineAndCheckCollisionValidity2(eval(currConsToPass),dnnOutput, triangle, currGroup, currDepth+1,0)
                       
                       
                        smallCubeCount += 1

                        if smallCubeReturn == 1:
                            # print("It is a valid collision,")
                            isValidCollision = 1
                            for process in processes:
                                os.kill(process.pid, signal.SIGKILL) 


                            return 1 
                        elif smallCubeReturn == 4:
                            # print("child process created")
                            pass
                            
                        else:
                            # print("Small cube has some non matching dnn output")
                            pass




                        # currSmallCubeDnnoutputs = intervalImageP3_4.computeIntervalImage_P3(currSmallCubeName, eval(currConsToPass),
                        #                                     nextRegionMinMaxValues, cornerPointsOfProjection, fromSplitRegion=1)    





                        # print("possible dnn outputs currSmallCubeDnnoutputs = ", currSmallCubeDnnoutputs)
                        # sleep(2)
                        # # currSmallCubeDnnoutputs = singleTriangleInvRegions30.computePixelIntervals(currSmallCubeName, And(intervalCons,currentRegionCons), fromSplitRegion=1)

                        #  intervalImageP3_3.computeIntervalImage_P3(nextGroupName, tempFromula,
                        #                    nextRegionMinMaxValues, nextRegionCornerPoints)
                        
                        
                        
                        
                        
                        
                        # if len(currSmallCubeDnnoutputs) == 1 and int(dnnOutput) != int(currSmallCubeDnnoutputs[0]):
                        #     print("Dnn output of small cube does not match ")
                        #     continue
                        # elif len(currSmallCubeDnnoutputs) == 1 and int(dnnOutput) == int(currSmallCubeDnnoutputs[0]):
                        #     print("Small cube has only one dnn output and it is matching with dnnOutput")
                        #     print("Backpropagate the region into previous multichild parent")
        
                        #     currentNodeString = str(currGroup)
                        #     currentNode = anytree.find(environment.A, filter_=lambda node: node.name==currentNodeString)
                        #     ancNodeWithMultChild ="" 
                        #     pathLengthToAncestor = 0
                        #     # if(int(numOfCurrNodeChilds) == 1):
                        #     ancNodeWithMultChild, pathLengthToAncestor  = find_ancestor_with_multiple_children(currentNode)
                        #     if(ancNodeWithMultChild == None):
                        #         print("It is a valid collision,")
                        #         return 1            
                                
                        #     print("ancNodeWithMultChild = ", ancNodeWithMultChild)
                        #     print("pathLengthToAncestor = ", pathLengthToAncestor)
                        #     # sleep(5)
                        #     #bakcpropagate the region
                        #     # currentRegionPath, currentRegionCons, currRegionOutputToCheck = backTrackARegionNStep(And(intervalCons,currentRegionCons),currGroup, pathLengthToAncestor )
                        #     currentRegionPath, currentRegionCons, currRegionOutputToCheck = backTrackARegionNStep(eval(currConsToPass),currGroup, pathLengthToAncestor )
                        
                        #     #refine the backtracked region
                        #     # smallCubeReturn = refineAndCheckCollisionValidity(And(intervalCons,currentRegionCons),currRegionOutputToCheck, triangle, currentRegionPath, 0)
                        #     smallCubeReturn = refineAndCheckCollisionValidity2(eval(currConsToPass),currRegionOutputToCheck, triangle, currentRegionPath, 0,1)
                            
                        #     if smallCubeReturn == 1:
                        #         print("It is a valid collision,")
                        #         return 1 
                        #     else:
                        #         print("Small cube has some non matching dnn output, @ ancestors")          
                        # else:
                        #     print("Refine the small cube again,.. again!!!!!")
                            
                        #     print("Small cube region cons = ", simplify(eval(currConsToPass)))
                        #     sleep(2)
                        #     # smallCubeReturn = refineAndCheckCollisionValidity(And(intervalCons,currentRegionCons),dnnOutput, triangle, currGroup, currDepth+1)
                        #     smallCubeReturn = refineAndCheckCollisionValidity2(eval(currConsToPass),dnnOutput, triangle, currGroup, currDepth+1,0)
                        #     if smallCubeReturn == 1:
                        #         print("It is a valid collision,")
                        #         return 1 
                        #     else:
                        #         print("Small cube has some non matching dnn output")
                    else:
                        # print("small cube is outside the intersection region.")
                        pass
    
    
    return 10                   
                        
                    
               
        
    
    
def handleMultipleLevelOfRefinement(triangle):
    
    global numberOfProcessCreated
    global uniqueIdList
    global processes
    global isValidCollision
    
    
    xp0, yp0, zp0 = Reals('xp0 yp0 zp0')

    dataFromProcessing = []
    for k in range(0, numberOfProcessCreated):
        fileName = str("tempFiles/")+str(uniqueIdList[k])+".txt"
        currFile = open(fileName,'r')
        currDataFromFile = []
        # print("\n\n",fileName)
        for i in range(6):
            line = currFile.readline()
            if line:  # Check if line is not empty
                # print(line.strip())  # Remove leading/trailing whitespace
                currDataFromFile.append(line.strip())
            else:
                break  # Exit if there are less than three lines
        foundARegion = int(currDataFromFile[2])
      
        if foundARegion == 1:
            # numOfRegionsFound = int(currDataFromFile[3])
            # regionConsSetString = currFile.readline()
            #here we convert the string to list of lists and then to z3 expression
            # currDataFromFile.append(regionConsSetString)
            dataFromProcessing.append(currDataFromFile)
        currFile.close()
        # print("dataFromProcessing ",dataFromProcessing)

    if len(dataFromProcessing) != 0:
        # print("len of dataToFromProcessing = ", len(dataFromProcessing)) 
        for i in range(0,len(dataFromProcessing)):
            currFileDataToBackTrack = dataFromProcessing[i]
            # print("currFileDataToBackTrack = ", currFileDataToBackTrack)
            
            currGroupToBacktrackName = currFileDataToBackTrack[4]
            
            backTrackRegionConsesString = str(currFileDataToBackTrack[5])
            # print("backTrackRegionConsesString = ", backTrackRegionConsesString)
            # print("number of regions to backtrack = ", len(backTrackRegionConsesString))
            list_of_lists = ast.literal_eval(backTrackRegionConsesString)

            # Print the result
            # print(list_of_lists)
            # print(len(list_of_lists)) 
            
            regionConsZ3ToBackTrack = []   
            
            for j in range(0, len(list_of_lists)):
                currSmallRegionToBack = list_of_lists[i]
                currRegionCons = And(True)
                # print(len(currSmallRegionToBack))
                for l in range(0,len(currSmallRegionToBack)):
                    currRegionCons = And(currRegionCons, eval(currSmallRegionToBack[l]))
                regionConsZ3ToBackTrack.append(currRegionCons)

            
            currentNodeString = str(currGroupToBacktrackName)
            currentNode = anytree.find(environment.A, filter_=lambda node: node.name==currentNodeString)
            ancNodeWithMultChild ="" 
            pathLengthToAncestor = 0
            # if(int(numOfCurrNodeChilds) == 1):
            ancNodeWithMultChild, pathLengthToAncestor  = find_ancestor_with_multiple_children(currentNode)
            if(ancNodeWithMultChild == None):
                # print("It is a valid collision,")
                # print("True collision detected !!!!!#2212###!!!!!")
                isValidCollision = 1                
                for process in processes:
                    os.kill(process.pid, signal.SIGKILL)                 
                return 1            
                
            # print("ancNodeWithMultChild = ", ancNodeWithMultChild)
            # print("pathLengthToAncestor = ", pathLengthToAncestor)
            
            for j in range(0, len(regionConsZ3ToBackTrack)):
                intersectionRegionConZ3_2 = regionConsZ3ToBackTrack[j]
                currentRegionPath, currentRegionCons, currRegionOutputToCheck = backTrackARegionNStep(intersectionRegionConZ3_2,currentNodeString, pathLengthToAncestor )
                # print(currentRegionPath, currRegionOutputToCheck)
                # print("\n\n")
                
                processes.clear()
                numberOfProcessCreated = 0
                uniqueIdList.clear()
                # print("Going to invoke refine function again")
                #refine the backtracked region
                refineAndCheckCollisionValidity2(currentRegionCons,currRegionOutputToCheck, triangle, currentRegionPath, 0,1)
            
                # print("processes ", processes)
                # Ensure all processes have finished
                # print("waiting for all processes to finish")
                for process in processes:
                    process.join()

                # print("All processes have finished")
                
                if isValidCollision == 1:
                    # print("True collision detected !!!!@@!#!#!!!")
                    return 1
                
                handleMultipleLevelOfRefinement(triangle)
                

    return 0


def checkValidityOfCollision(currGroup, triangle,currRegionPPL,dnnOutput,pd5):
    # print("\n\n reached checkValidity of collision function")
    # print("curr Triangle = ",triangle)
    # print("curr region = ", currRegionPPL)
    # print("dnn output = ", dnnOutput)
    # print("currGroup =", currGroup)
    
    global smallCubeCount
    global processes
    global uniqueIdList
    global num_processes
    global numberOfProcessCreated
    smallCubeCount =0

    processes.clear()
    numberOfProcessCreated = 0
    uniqueIdList.clear()
    num_processes = 30
    total_tasks = 100
        
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
    
 
    validityCheckStartTime = datetime.now()
    # print("validityCheckStartTime = ", validityCheckStartTime)
    
    currentNodeString = str(currGroup)
    currentNode = anytree.find(environment.A, filter_=lambda node: node.name==currentNodeString)
    numOfCurrNodeChilds = len(currentNode.children)
    # print("numOfCurrNodeChilds = ",numOfCurrNodeChilds)
    
    ancNodeWithMultChild ="" 
    pathLengthToAncestor = 0
    if(int(numOfCurrNodeChilds) == 1):
        ancNodeWithMultChild, pathLengthToAncestor  = find_ancestor_with_multiple_children(currentNode)
        if(ancNodeWithMultChild == None):
            # print("It is a valid collision,")
            return 1, And(True)
        
        
    # print("ancNodeWithMultChild = ", ancNodeWithMultChild)
    # print("pathLengthToAncestor = ", pathLengthToAncestor)
    
    
    
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


    # x0 = int(x0*pow(10,7)) 
    # y0 = int(y0*pow(10,7)) 
    # z0 = int(z0*pow(10,7)) 
    
    # x1 = int(x1*pow(10,7)) 
    # y1 = int(y1*pow(10,7)) 
    # z1 = int(z1*pow(10,7)) 

    # x2 = int(x2*pow(10,7)) 
    # y2 = int(y2*pow(10,7)) 
    # z2 = int(z2*pow(10,7)) 

    x0 = int(x0*pow(10,3)) 
    y0 = int(y0*pow(10,3)) 
    z0 = int(z0*pow(10,3)) 
    
    x1 = int(x1*pow(10,3)) 
    y1 = int(y1*pow(10,3)) 
    z1 = int(z1*pow(10,3)) 

    x2 = int(x2*pow(10,3)) 
    y2 = int(y2*pow(10,3)) 
    z2 = int(z2*pow(10,3)) 
    
    # trianglePolyhedron = NNC_Polyhedron(3,'empty')
    # trianglePolyhedron.add_generator(point( x0*xp0+y0*yp0+z0*zp0 ))
    # trianglePolyhedron.add_generator(point( x1*xp0+y1*yp0+z1*zp0 ))
    # trianglePolyhedron.add_generator(point( x2*xp0+y2*yp0+z2*zp0 ))
    # trianglePolyhedron = NNC_Polyhedron(3,'empty')
    # trianglePolyhedron.add_generator(point( x0*xp0+y0*yp0+z0*zp0, pow(10,7) ))
    # trianglePolyhedron.add_generator(point( x1*xp0+y1*yp0+z1*zp0, pow(10,7) ))
    # trianglePolyhedron.add_generator(point( x2*xp0+y2*yp0+z2*zp0, pow(10,7) ))

    trianglePolyhedron = NNC_Polyhedron(3,'empty')
    trianglePolyhedron.add_generator(point( x0*xp0+y0*yp0+z0*zp0, pow(10,3) ))
    trianglePolyhedron.add_generator(point( x1*xp0+y1*yp0+z1*zp0, pow(10,3) ))
    trianglePolyhedron.add_generator(point( x2*xp0+y2*yp0+z2*zp0, pow(10,3) ))
    
    
    # print("\n\n\n trianglePolyhedron.minimized_constraints()")
    # print(trianglePolyhedron.constraints())
    # print(trianglePolyhedron.minimized_constraints())
    
    
    # print("path Hull Constraint")
    # print(pd5.minimized_constraints())
    
    
    # #find the exact intersecting region
    pd5.intersection_assign(trianglePolyhedron)
    # print("triangle hull intersection region :", pd5.minimized_constraints()) 
    # print("\n")
    
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
    # print("touching file")
    os.system(tempstring)

    # tempstring = "gcc pplTrianglePath.cpp -o pplTrianglePath -L/home2/habeebp/opt/include/ -L/home2/habeebp/opt/lib/ -I/home2/habeebp/opt/include/ -lstdc++ -lppl -lgmpxx -lgmp"
    tempstring = "gcc pplTrianglePath.cpp -o pplTrianglePath -L/home/habeeb/projectFinal/softwares/ppl-1.2/include/ -L/home/habeeb/projectFinal/softwares/ppl-1.2/lib/ -I/home/habeeb/projectFinal/softwares/ppl-1.2/include/ -lstdc++ -lppl -lgmpxx -lgmp"
    
    # print("compiling pplTrianglePath.cpp")
    os.system(tempstring)

    tempstring = "./pplTrianglePath"
    # os.system(tempstring)

    pplOutputFilePtr = open("triangleHullRegionpolyhedron.txt",'r')
    # print("\n\n From ppl\n")
    preRegionpolyhedronConString = pplOutputFilePtr.read()
    # print(preRegionpolyhedronConString)
    pplOutputFilePtr.close()
    
    preRegionpolyhedronConString = str(preRegionpolyhedronConString)
    preRegionpolyhedronConString = preRegionpolyhedronConString.replace("A","xp0")
    preRegionpolyhedronConString = preRegionpolyhedronConString.replace("B","yp0")
    preRegionpolyhedronConString = preRegionpolyhedronConString.replace("C","zp0")
    preRegionpolyhedronConString = preRegionpolyhedronConString.replace(" = ","==")
    preRegionpolyhedronConStringList = preRegionpolyhedronConString.split(",")
    
    
    # print(preRegionpolyhedronConStringList)
    
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
    # print("Current image cons = ", currGroupCons)
    
    currImageConsZ3 = currGroupCons   
    
    
    intersectionRegionConZ3 = And(currImageConsZ3,newCons)
    
    # print("Final Intersection region to back track =")
    # print(simplify(intersectionRegionConZ3))
    # print("\n\nsimplified formula : ", simplify(intersectionRegionConZ3))
    
    
    xp0,yp0,zp0 = Reals('xp0 yp0 zp0')
    xp1,yp1,zp1 = Reals('xp1 yp1 zp1')
    currentRegionPath =  currGroup
    currentRegionCons = intersectionRegionConZ3
    currRegionOutputToCheck = dnnOutput


    # ss100 = Solver()
    # ss100.add(simplify(currentRegionCons))
    # print("number of cons = ", len(ss100.assertions()))
    # print(ss100.check())
    

    if numOfCurrNodeChilds == 1:
        
        # print("Back propagating the final region/node to multi child parent node")
        # sleep(2)
        currentRegionPath, currentRegionCons, currRegionOutputToCheck = backTrackARegionNStep(intersectionRegionConZ3,currGroup, pathLengthToAncestor )
    
        # print("Region to refine\n", currentRegionCons)
        # print("simplified = ",simplify(currentRegionCons))
        # print("currentRegionPath = ", currentRegionPath)
        # # dnnOutputToCheck = int(currentRegionPath[currentRegionPath.rfind("_")+1:])
        # print("Current region dnn out to check = ",currRegionOutputToCheck)
        
    # sleep(10)
    
    

    refineAndCheckCollisionValidity2(currentRegionCons,currRegionOutputToCheck, triangle, currentRegionPath,0,0)



    # print("processes ", processes)
    # Ensure all processes have finished
    # print("waiting for all processes to finish")
    for process in processes:
        process.join()


    # print("All processes have finished")
    # print("time taken ", datetime.now()-validityCheckStartTime)

    


   

    # print("time taken @end ", datetime.now()-validityCheckStartTime)
    if isValidCollision == 1:
        # print("True collision detected !!!!!!!")
        return 1, intersectionRegionConZ3
    
    # handleMultipleLevelOfRefinement(triangle)
    # print("Skipping Handle multilevel")

    # print("time taken @end ", datetime.now()-validityCheckStartTime)
    if isValidCollision == 1:
        # print("True collision detected !!!!!!!")
        return 1, intersectionRegionConZ3
    else:
        # print("Not a valid collision")
        return 0, intersectionRegionConZ3

    # if refineStatus ==1:
    #     print("True collision detected !!!!!!!")
    #     return 1
    # elif(refineStatus == 0):
    #     print("Not a valid collision")
    #     return 0
    
    
    # sleep(100)
                    
    
    
 
    
        
    
    
    
    







