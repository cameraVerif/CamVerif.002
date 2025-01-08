
#libraries
from z3 import *
from time import time
from datetime import datetime 
from collections import Counter
import numpy as np
import itertools
from time import sleep


import commonFunctions_1
import environment
import camera
import scene
import pyparma_posInvRegion40
import oldInvComputation1
import pythonRenderAnImage2


import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import cv2 
from tensorflow.keras.models import load_model


import cv2
import onnx
import onnxruntime

from onnx import numpy_helper



imageWidth = camera.imageWidth
imageHeight = camera.imageHeight
mProj = commonFunctions_1.mProj

xp0, yp0, zp0 = Reals('xp0 yp0 zp0')

globalInsideVertexDataToPPL =[]
globalIntersectingVertexDataToPPL = []
edges = []


outValues = [0]*scene.numOfVertices*4*5 

def getDNNOutput(inputImage):
    model = load_model('saved_models/3_2')
    # print("inputImage = ",inputImage)
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




def getDNNOutput_onnx(inputImage,networkName):
    dnnOutput = 1
    # model = onnx.load('iisc_net1.onnx')

    image = cv2.imread(inputImage)    
    image = cv2.resize(image, (49, 49)).copy()

    a, b, c = image.shape
    image = image.reshape(1, a, b, c)
    print(image.shape)

    image = image.astype(np.float32) / 255.0
    # image2 = tf.convert_to_tensor(image)

    session = onnxruntime.InferenceSession(networkName)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # print(input_name)
    # print(output_name)

    result = session.run([output_name], {input_name: image})
    dnnOutput  = np.argmax(np.array(result).squeeze(), axis=0)
    print("networkName = ", networkName)
    print("dnnOutput = ", dnnOutput)
    return dnnOutput 


def computeOutcodeAtPos(i,outcodeP0, inx, iny,inz):
    

    mProj = commonFunctions_1.mProj

    outcode = 0   
    outx   = inx * mProj[0][0] + iny * mProj[1][0] + inz * mProj[2][0] +  mProj[3][0]
    outy   = inx * mProj[0][1] + iny * mProj[1][1] + inz * mProj[2][1] +  mProj[3][1] 
    outz   = inx * mProj[0][2] + iny * mProj[1][2] + inz * mProj[2][2] +  mProj[3][2] 
    w      = inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3] 
    
    global outValues
    outValues[i*4+0] = outx
    outValues[i*4+1] = outy
    outValues[i*4+2] = outz
    outValues[i*4+3] = w
    
    outValueToReturn = [outx, outy, outz]
    # print(inx, iny, inz, outx, outy, outz,w)
    # print(mProj)
	
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
        
    return outValueToReturn, w


def pixelValue(point, w):
    t0 = point[0]/w
    t1 = point[1]/w
    t2 = point[2]/w
    
    # print(((t0 + 1) * 0.5 * imageWidth),((1 - (t1 + 1) * 0.5) * imageHeight), t2)
    originalPixel = [int((t0 + 1) * 0.5 * imageWidth),int((1 - (t1 + 1) * 0.5) * imageHeight), t2]
    # print("pixel value before min = ",originalPixel  )
    
    raster0 = min(imageWidth - 1, int((t0 + 1) * 0.5 * imageWidth))
    raster1 = min(imageHeight - 1, int((1 - (t1 + 1) * 0.5) * imageHeight))
    raster2 = t2
    
    
    a2 = str((t0 + 1) * 0.5 * imageWidth)
    b2 =str((1 - (t1 + 1) * 0.5) * imageHeight)
     
       
        
    if "." in a2:
        ta1 = int(a2.split(".")[0])
        ta2 = "."+a2.split(".")[1]
    else:
        ta1 = int(a2)
        ta2 = 0
        
    if "." in b2:                    
        tb1 = int(b2.split(".")[0])
        tb2 = "."+b2.split(".")[1]
    else:
        tb1 = int(b2)
        tb2 = 0

    currPixels = []

    currPixels.append(int(ta1))
    currPixels.append(int(tb1))
    
    if (float(ta2) > 0.999):
        # ta1 = int(ta1)+1
        currPixels.append(int(ta1)+1)
        currPixels.append(int(tb1))
    elif(float(ta2) < .0001 ):
        currPixels.append(int(ta1)-1)
        currPixels.append(int(tb1))
            
    if (float(tb2) > 0.999 ):              
        currPixels.append(int(ta1))
        currPixels.append(int(tb1)+1)
            
    elif(float(tb2) < .0001 ):
        currPixels.append(int(ta1))
        currPixels.append(int(tb1)-1)

    
    return currPixels,originalPixel 


def findEdges(currVetex):
    flag = 0
    edge1 = 0
    edge2 = 0
    
    for currEdge in edges:
        if(currVetex == currEdge[0] or currVetex == currEdge[1]):
            if(flag == 0):
                if (currVetex != currEdge[0]):
                    edge1 = currEdge[0]
                else:
                    edge1 = currEdge[1]
                flag = 1
            else:
                if (currVetex != currEdge[0]):
                    edge2 = currEdge[0]
                else:
                    edge2 = currEdge[1]
    return edge1, edge2



def removeEdges(outsideVertex):
    #TODO: we can rewrite this function, 
    numberOfOccur = 0
    
    for currEdge in edges:
        if(currEdge[0] == outsideVertex or currEdge[1] == outsideVertex):
            numberOfOccur+=1
    
    for i in range(0,numberOfOccur):
        for currEdge in edges:
            if(currEdge[0] == outsideVertex or currEdge[1] == outsideVertex):
                edges.remove(currEdge)
                break

def cpp_vertexPlaneIntersectionPoint(insideVertex, outsideVertex, insidevertexW, outsideVertexW, intersectionPlane):

    x1 = insideVertex[0]
    y1 = insideVertex[1]
    z1 = insideVertex[2]
    w1 = insidevertexW

    x2 = outsideVertex[0]
    y2 = outsideVertex[1]
    z2 = outsideVertex[2]
    w2 = outsideVertexW

    #3 => w=-x , 2 => w=x, 1 => w=-y, 0=> w=y, 5 ==>w=-z near, 4==>w=far
    if(intersectionPlane == 3):
        t1= (-w1-x1)/(w2-w1+x2-x1)
    elif(intersectionPlane == 2):
        t1= (x1-w1)/(w2-w1-x2+x1)
    elif(intersectionPlane == 1):
        t1 = (-w1-y1)/(w2-w1+y2-y1)
    elif(intersectionPlane == 0):
        t1 = (-w1+y1)/(w2-w1-y2+y1)
    elif(intersectionPlane == 5):
        t1 = (-1-w1)/(w2-w1)
    elif(intersectionPlane == 4):   
        t1 = (-1000-w1)/(w2-w1)
        
        
    intersectionPoint = [x1+t1*(x2-x1), y1+ t1*(y2-y1), z1+t1*(z2-z1)]
    intersectionPointW = w1+t1*(w2-w1)
    # print("intersectionPoint = ",intersectionPoint)
    

    return t1, intersectionPoint, intersectionPointW 

def clipCoordinateToOutcode(out,  w):
    
    outcode = [0]*6
    if( not (-w <= out[0]) ):
        outcode[3] = 1
    elif(not (out[0] <=w) ):
        outcode[2] = 1
        
    if( not (-w <= out[1]) ):
        outcode[1] = 1
    elif(not (out[1] <=w) ):
        outcode[0] = 1
        
    if( not(-w <= out[2]) ):
        outcode[5] = 1
    elif(not (out[2] <=w) ):
        outcode[4] = 1
    
    return outcode        
  
def getVertexPixelValueIntersectZ3(x,y,z, plane):
    s3 = Solver()
    set_param('parallel.enable', True)
    set_option(rational_to_decimal=True)
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
    if s3.check() == sat:
        m = s3.model()
        #print("etVertexPixelValueIntersectZ3: model from solver :",m)
        # a1 = str(eval("m[a].numerator_as_long()/m[a].denominator_as_long()"))
        # b1 = str(eval("m[b].numerator_as_long()/m[b].denominator_as_long()"))
        # # print(a1,b1)   
        # a1= float(a1)
        # b1= float(b1)
        
        a2 = str(eval("m[a]")).replace("?","")
        b2 = str(eval("m[b]")).replace("?","")
        # print("a2 = ", a2)
        # print("b2 = ", b2)
        
        # a2 = a2.split(".")[0]
        # b2 = b2.split(".")[0]
        # print("a2 = ", a2)
        # print("b2 = ", b2)
        
        if "." in a2:
            ta1 = int(a2.split(".")[0])
            ta2 = "."+a2.split(".")[1]
        else:
            ta1 = int(a2)
            ta2 = 0
            
        if "." in b2:                    
            tb1 = int(b2.split(".")[0])
            tb2 = "."+b2.split(".")[1]
        else:
            tb1 = int(b2)
            tb2 = 0
        
        # print("fraction of a = ", float(str(ta2).replace("?","")))
        # print("fraction of b ", float(str(tb2).replace("?","")))


        if plane ==0 :
            tb1 = 0
        elif plane ==1 :
            tb1 = 49
        elif plane ==2 :
            ta1 = 49
        elif plane ==3 :
            ta1 = 0



        currPixels = []

        currPixels.append(int(ta1))
        currPixels.append(int(tb1))
        
        # if (float(ta2) > 0.999 and plane !=2):
        #     # ta1 = int(ta1)+1
        #     currPixels.append(int(ta1)+1)
        #     currPixels.append(int(tb1))
        # elif(float(ta2) < .0001 and plane !=3):
        #     currPixels.append(int(ta1)-1)
        #     currPixels.append(int(tb1))
                
        # if (float(tb2) > 0.999 and plane !=1):              
        #     currPixels.append(int(ta1))
        #     currPixels.append(int(tb1)+1)
              
        # elif(float(tb2) < .0001 and plane !=0):
        #     currPixels.append(int(ta1))
        #     currPixels.append(int(tb1)-1)


        if (float(ta2) > 0.999 ):
            # ta1 = int(ta1)+1
            currPixels.append(int(ta1)+1)
            currPixels.append(int(tb1))
        elif(float(ta2) < .0001 ):
            currPixels.append(int(ta1)-1)
            currPixels.append(int(tb1))
                
        if (float(tb2) > 0.999 ):              
            currPixels.append(int(ta1))
            currPixels.append(int(tb1)+1)
              
        elif(float(tb2) < .0001 ):
            currPixels.append(int(ta1))
            currPixels.append(int(tb1)-1)



        


        return currPixels
    else:
        print("no sat image")
        p =[-1,-1]
        return p

def planeEdgeIntersectionUpdated(plane,insideVertex, outsideVertex,m, newCode = 0): 
    # print("____________________")
    # print(" Inside planeEdgeIntersection Updated function")
    # print(plane,insideVertex, outsideVertex)
    s1 = Solver()
    set_param('parallel.enable', True)
    # set_option(rational_to_decimal=False)
    # set_option(precision=20)
    set_param('parallel.enable', True)
    s1.set("sat.local_search_threads", 26)
    s1.set("sat.threads", 26)
    s1.set("timeout",10000)
    
   
    #print("inside vertex :", insideVertex)
    #print("outside vertex :", outsideVertex)
    #print(posXp,posYp,posZp)
    # print("m[xp0] = ", m[xp0])
    # print("m[yp0] = ", m[yp0])
    # print("m[zp0] = ", m[zp0])
    xv0 = 0
    yv0 = 0
    zv0 = 0
    
    
    if newCode == 0:    
        xv0 = vertices[insideVertex*3+0] - m[xp0]
        yv0 = vertices[insideVertex*3+1] - m[yp0]
        zv0 = vertices[insideVertex*3+2] - m[zp0]
        wv0 = -(vertices[insideVertex*3+2] - m[zp0])
        
        xv1 = vertices[outsideVertex*3+0] - m[xp0]
        yv1 = vertices[outsideVertex*3+1] - m[yp0]
        zv1 = vertices[outsideVertex*3+2] - m[zp0]
        wv1 =  - (vertices[outsideVertex*3+2] - m[zp0])
    
    else:
        
        xv0 = insideVertex[0] - m[xp0]
        yv0 = insideVertex[1] - m[yp0]
        zv0 = insideVertex[2] - m[zp0]
        
        xv1 = outsideVertex[0] - m[xp0]
        yv1 = outsideVertex[1] - m[yp0]
        zv1 = outsideVertex[2] - m[zp0]
                
    
    # print(xv0,yv0,zv0, xv1,yv1,zv1)
   
    
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
    # s1.check()
    
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
        # dummyValue = -1000
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
    # s1.check()
    # if(s1.check() ==sat):
    #     m23 =s1.model()
    #     print(m23)

    s1.add(xl == xk)
    s1.add(yl == yk)
    s1.add(zl == zk)
    
    # for c in s1.assertions():
    #     print(c,"\n")
    
    result = s1.check()
    if(result ==sat):
        # print("intersecting with the plane using linear combination; and z3")
        m2 = s1.model()
        # print(m2)
        
        insideFraction = eval("m2[p].numerator_as_long()/m2[p].denominator_as_long() " )
        outsideFraction = eval("m2[q].numerator_as_long()/m2[q].denominator_as_long() " )
        
         
       
        intersectionPoint = [0,0,0,0]
       
        
        vertexPixelValue2 = getVertexPixelValueIntersectZ3(m2[p]*xv0+m2[q]*xv1,\
                                                           m2[p]*yv0+m2[q]*yv1,\
                                                           m2[p]*zv0+m2[q]*zv1, plane) 
        
        intersectionPoint = [0,0,0,0]
        if newCode == 0:
            xv0 = vertices[insideVertex*3+0] 
            yv0 = vertices[insideVertex*3+1] 
            zv0 = vertices[insideVertex*3+2] 
            wv0 = -(vertices[insideVertex*3+2] )
                            
            xv1 = vertices[outsideVertex*3+0] 
            yv1 = vertices[outsideVertex*3+1] 
            zv1 = vertices[outsideVertex*3+2]
            wv1 =  - (vertices[outsideVertex*3+2] )
            
            intersectionPoint[0] =  eval("(1- outsideFraction)*xv0+ outsideFraction*xv1")
            intersectionPoint[1] = eval("(1- outsideFraction)*yv0+ outsideFraction*yv1")
            intersectionPoint[2] = eval("(1- outsideFraction)*zv0+ outsideFraction*zv1")
            intersectionPoint[3] = eval("(1- outsideFraction)*wv0+ outsideFraction*wv1")
        else:
            
            xv0 = insideVertex[0] 
            yv0 = insideVertex[1] 
            zv0 = insideVertex[2] 
            
            xv1 = outsideVertex[0] 
            yv1 = outsideVertex[1] 
            zv1 = outsideVertex[2] 
        
            intersectionPoint[0] =  eval("(1- outsideFraction)*xv0+ outsideFraction*xv1")
            intersectionPoint[1] = eval("(1- outsideFraction)*yv0+ outsideFraction*yv1")
            intersectionPoint[2] = eval("(1- outsideFraction)*zv0+ outsideFraction*zv1")
            
       
        
        return 1 , vertexPixelValue2, intersectionPoint,m2[p],m2[q]
    elif result == unsat:
        # 
        # print("Returning planeEdgeIntersection")
        return 0 , [-1,-1],[0,0,0],-1,-1  
    
def outValueToWorldCoordinates(out):

    x = out[0]
    y = out[1]
    z = out[2]
    
    a = np.array([
        [mProj[0][0], mProj[1][0], mProj[2][0]],
        [mProj[0][1], mProj[1][1], mProj[2][1]],
        [mProj[0][2], mProj[1][2], mProj[2][2]]
        ])
    
    b = np.array([x-mProj[3][0],y-mProj[3][1],z-mProj[3][2]])
    
    sol = np.linalg.solve(a,b)
    
    return sol
    


def getInvariantRegion(currTriangle, posXp, posYp, posZp, m,currGroupName):
    # print("Invariant Region Computation Started")

    nvertices = commonFunctions_1.getNVertices()
    vertices = commonFunctions_1.getVertices()
    vertColours = commonFunctions_1.getVertColours()
    tedges = commonFunctions_1.getTEdges()

    depthOfTheInitialCube = environment.depthOfTheInitialCube

    
    
    #initialize the environment
    # print("posXp: ", posXp)
    # print("posYp: ", posYp)
    # print("posZp: ", posZp)

    invariantRegion = []

    #read scene data
    numOfTriangles, numOfVertices, numOfEdges = commonFunctions_1.getSceneData()

    
    # print("currTriangle: ", currTriangle)

    #get the vertices of the triangle
    # triangleVertices = commonFunctions_1.getTriangleVertices(currTraingle)
    # print("triangleVertices: ", triangleVertices)

    vertex0 = nvertices[currTriangle*3+0]
    vertex1 = nvertices[currTriangle*3+1]
    vertex2 = nvertices[currTriangle*3+2]
    currTriangleVertices = [vertex0, vertex1,vertex2]
    
    v0Vertex = [vertices[currTriangleVertices[0]*3+0], vertices[currTriangleVertices[0]*3+1],vertices[currTriangleVertices[0]*3+2] ]
    v1Vertex = [vertices[currTriangleVertices[1]*3+0], vertices[currTriangleVertices[1]*3+1],vertices[currTriangleVertices[1]*3+2] ]
    v2Vertex = [vertices[currTriangleVertices[2]*3+0], vertices[currTriangleVertices[2]*3+1],vertices[currTriangleVertices[2]*3+2] ]


    if Counter(v0Vertex) == Counter(v1Vertex) or Counter(v0Vertex) == Counter(v2Vertex) or Counter(v1Vertex) == Counter(v2Vertex):       
        return 0, And(True)


    if(vertices[currTriangleVertices[0]*3+2] > posZp-1 + depthOfTheInitialCube and 
        vertices[currTriangleVertices[1]*3+2] > posZp-1 + depthOfTheInitialCube and 
        vertices[currTriangleVertices[2]*3+2] > posZp-1 + depthOfTheInitialCube ):
        return 0, And(True)
    
    if tedges[(currTriangle) *6+0] == vertex0 :
            edge0_v0 = 0
    elif tedges[(currTriangle) *6+0] == vertex1 :
        edge0_v0 = 1
    else:
        edge0_v0 = 2
    
    if tedges[ (currTriangle) *6+1] == vertex0 :
        edge0_v1 = 0
    elif tedges[(currTriangle) *6+1] == vertex1 :
        edge0_v1 = 1
    else:
        edge0_v1 = 2
        
        
    if tedges[ (currTriangle) *6+2] == vertex0 :
        edge1_v0 = 0
    elif tedges[ (currTriangle) *6+2] == vertex1 :
        edge1_v0 = 1
    else:
        edge1_v0 = 2
        
    if tedges[ (currTriangle) *6+3] == vertex0 :
        edge1_v1 = 0
    elif tedges[ (currTriangle) *6+3] == vertex1 :
        edge1_v1 = 1
    else:
        edge1_v1 = 2
        
    
    if tedges[ (currTriangle) *6+4] == vertex0 :
        edge2_v0 = 0
    elif tedges[ (currTriangle) *6+4] == vertex1 :
        edge2_v0 = 1
    else:
        edge2_v0 = 2 
        
    if tedges[(currTriangle) *6+5] == vertex0 :
        edge2_v1 = 0
    elif tedges[ (currTriangle) *6+5] == vertex1 :
        edge2_v1 = 1
    else:
        edge2_v1 = 2  

    edgeVertexIndices = [edge0_v0, edge0_v1,edge1_v0, edge1_v1, edge2_v0, edge2_v1 ]

    print("edgeVertexIndices: ", edgeVertexIndices)

    globalInsideVertexDataToPPL.clear()
    globalIntersectingVertexDataToPPL.clear()

    insideVertexDetailsToPPL = [] #store vertex index number,xpixel,ypixel
    numberOfFullyInsideVertices = 0
    numberOfIntersectingEdges = 0
    intersectingEdgeDataToPPL = []

    fullyInsideVerticesNumber = []
    intersectingVerticesNumber = []

    intersectingData = dict()
    intersectingData.clear()

    global edges
    edges.clear()
        
    tr_num_of_vertices = 3
    tr_curr_num_of_vertex = 3
    tr_vertex_coordinates = np.zeros((100,3))
    tr_vertex_ws = [0]*100
    tr_vertex_outcodes = np.zeros((100,6))
    tr_vertices_set = []


    notTheCurrentPosCons1 = Or(xp0!= m[xp0], yp0!=m[yp0],zp0!= m[zp0])
    currentPlane = 0 #top plane
    
    outcodeP0 = [0]*30*6
        
    
    outValue0, outW0 = computeOutcodeAtPos(0,outcodeP0, 
                        vertices[currTriangleVertices[0]*3+0]-posXp ,
                        vertices[currTriangleVertices[0]*3+1]-posYp,
                        vertices[currTriangleVertices[0]*3+2]-posZp )
    outValue1, outW1 = computeOutcodeAtPos(1,outcodeP0, 
                        vertices[currTriangleVertices[1]*3+0]-posXp ,
                        vertices[currTriangleVertices[1]*3+1]-posYp,
                        
                        vertices[currTriangleVertices[1]*3+2]-posZp )
    
    outValue2, outW2 = computeOutcodeAtPos(2,outcodeP0, 
                        vertices[currTriangleVertices[2]*3+0]-posXp ,
                        vertices[currTriangleVertices[2]*3+1]-posYp,
                        vertices[currTriangleVertices[2]*3+2]-posZp )



    bit0 = outcodeP0[0] & outcodeP0[6] & outcodeP0[12]
    bit1 = outcodeP0[1] & outcodeP0[7] & outcodeP0[13]
    bit2 = outcodeP0[2] & outcodeP0[8] & outcodeP0[14]
    bit3 = outcodeP0[3] & outcodeP0[9] & outcodeP0[15]
    bit4 = outcodeP0[4] & outcodeP0[10] & outcodeP0[16]
    bit5 = outcodeP0[5] & outcodeP0[11] & outcodeP0[17]
    
    
    
    tr_vertex_outcodes[0] = outcodeP0[0:6]
    tr_vertex_outcodes[1] = outcodeP0[6:12]
    tr_vertex_outcodes[2] = outcodeP0[12:18]

    # print("outcode0: ", tr_vertex_outcodes[0])
    # print("outcode1: ", tr_vertex_outcodes[1])
    # print("outcode2: ", tr_vertex_outcodes[2])

    tr_vertex_coordinates[0] = outValue0
    tr_vertex_coordinates[1] = outValue1
    tr_vertex_coordinates[2] = outValue2

    tr_vertex_ws[0]=outW0
    tr_vertex_ws[1]=outW1
    tr_vertex_ws[2]=outW2


    tr_vertices_set.append(0)
    tr_vertices_set.append(1)
    tr_vertices_set.append(2)

    # edges.append([0,1])
    # edges.append([0,2])
    # edges.append([1,2])
    
    edges.append([edge0_v0,edge0_v1])
    edges.append([edge1_v0,edge1_v1])
    edges.append([edge2_v0,edge2_v1])

    currImageColours = dict()
    currImageColours.clear()
    
    currImageColours[0] = [scene.vertColours[currTriangleVertices[0]*3+0],
                            scene.vertColours[currTriangleVertices[0]*3+1],
                            scene.vertColours[currTriangleVertices[0]*3+2]] 
    
    currImageColours[1] = [scene.vertColours[currTriangleVertices[1]*3+0],
                            scene.vertColours[currTriangleVertices[1]*3+1],
                            scene.vertColours[currTriangleVertices[1]*3+2]] 
    
    currImageColours[2] = [scene.vertColours[currTriangleVertices[2]*3+0],
                            scene.vertColours[currTriangleVertices[2]*3+1],
                            scene.vertColours[currTriangleVertices[2]*3+2]] 



    if(not any(outcodeP0)):            
        for currVert in range(0,3):      
            x = vertices[currTriangleVertices[currVert]*3+0]
            y = vertices[currTriangleVertices[currVert]*3+1]
            z = vertices[currTriangleVertices[currVert]*3+2]
           
            if currVert == 0:
                vertexPixelValue, temp = pixelValue(outValue0,outW0)
            elif currVert == 1:
                vertexPixelValue, temp = pixelValue(outValue1,outW1)
            else:
                vertexPixelValue, temp = pixelValue(outValue2,outW2)
            
            # print("currVert: ", currVert)
            # print("vertexPixelValue: ", vertexPixelValue)
            fflag = 0
            currVertexPixelData = [] 
            for vPixels in range(0,len(vertexPixelValue),2):
                if(vertexPixelValue[vPixels] >=0 and vertexPixelValue[vPixels+1]>=0):
                    currVertColour = scene.vertColours[currTriangleVertices[currVert]*3:currTriangleVertices[currVert]*3+3]
                    tempPixelData = [currTriangleVertices[currVert], vertexPixelValue[vPixels],vertexPixelValue[vPixels+1],currVert,currVertColour]
                    insideVertexDetailsToPPL.append(tempPixelData)
                    currVertexPixelData.append(tempPixelData)                            
                    if fflag == 0:
                        numberOfFullyInsideVertices+=1
                        fflag = 1
                    # currImage.append([vertexPixelValue[vPixels],vertexPixelValue[vPixels+1]])
                    
                    # print("Adding data to PPL insideVertexDetailsToPPL")
            if fflag == 1:
                globalInsideVertexDataToPPL.append(currVertexPixelData)
                fullyInsideVerticesNumber.append(currVert)
    
    elif(bit0 or bit1 or bit2 or bit3 or bit4 or bit5):            
        
        return 0, And(True)
    else:
        
        #find data of the fully inside vertices
        vert0_pos = outcodeP0[0] | outcodeP0[1] | outcodeP0[2] | outcodeP0[3] | outcodeP0[4] | outcodeP0[5]
        vert1_pos = outcodeP0[6] | outcodeP0[7] | outcodeP0[8] | outcodeP0[9] | outcodeP0[10] | outcodeP0[11]
        vert2_pos = outcodeP0[12] | outcodeP0[13] | outcodeP0[14] | outcodeP0[15] | outcodeP0[16] | outcodeP0[17]
        
        # print("vert0_pos: ", vert0_pos)
        # print("vert1_pos: ", vert1_pos)
        # print("vert2_pos: ", vert2_pos)

        if(not vert0_pos):
            currVert =0
            
            
            vertexPixelValue, temp = pixelValue(outValue0,outW0)  
            
            # print("vertexPixelValue: ", vertexPixelValue)
            fflag = 0
            currVertexPixelData = [] 
            for vPixels in range(0,len(vertexPixelValue),2):
                if(vertexPixelValue[vPixels] >=0 and vertexPixelValue[vPixels+1]>=0):
                    currVertColour = scene.vertColours[currTriangleVertices[currVert]*3:currTriangleVertices[currVert]*3+3]
                    tempPixelData = [currTriangleVertices[currVert], vertexPixelValue[vPixels],vertexPixelValue[vPixels+1],currVert,currVertColour]
                    insideVertexDetailsToPPL.append(tempPixelData)
                    currVertexPixelData.append(tempPixelData)                            
                    if fflag == 0:
                        numberOfFullyInsideVertices+=1
                        fflag = 1
                    # currImage.append([vertexPixelValue[vPixels],vertexPixelValue[vPixels+1]])
                   
            if fflag == 1:
                globalInsideVertexDataToPPL.append(currVertexPixelData)
                fullyInsideVerticesNumber.append(currVert)
            
        
        if(not vert1_pos):
            currVert =1
            
            vertexPixelValue, temp = pixelValue(outValue1,outW1)   
            # print("vertexPixelValue: ", vertexPixelValue)
            fflag = 0
            currVertexPixelData = [] 
            for vPixels in range(0,len(vertexPixelValue),2):
                if(vertexPixelValue[vPixels] >=0 and vertexPixelValue[vPixels+1]>=0):
                    currVertColour = scene.vertColours[currTriangleVertices[currVert]*3:currTriangleVertices[currVert]*3+3]
                    tempPixelData = [currTriangleVertices[currVert], vertexPixelValue[vPixels],vertexPixelValue[vPixels+1],currVert, currVertColour]
                    insideVertexDetailsToPPL.append(tempPixelData)
                    currVertexPixelData.append(tempPixelData)                            
                    if fflag == 0:
                        numberOfFullyInsideVertices+=1
                        fflag = 1
                    # currImage.append([vertexPixelValue[vPixels],vertexPixelValue[vPixels+1]])
                    # print("edge_v0: ", edge_v0," pixelvalue : ", vertexPixelValue)
                    # print("Adding data to PPL insideVertexDetailsToPPL")
            if fflag == 1:
                globalInsideVertexDataToPPL.append(currVertexPixelData)
                fullyInsideVerticesNumber.append(currVert)
        
        if(not vert2_pos):
            currVert =2
         
            vertexPixelValue, temp = pixelValue(outValue2,outW2)  
            # print("vertexPixelValue: ", vertexPixelValue)
            fflag = 0
            currVertexPixelData = [] 
            for vPixels in range(0,len(vertexPixelValue),2):
                if(vertexPixelValue[vPixels] >=0 and vertexPixelValue[vPixels+1]>=0):
                    currVertColour = scene.vertColours[currTriangleVertices[currVert]*3:currTriangleVertices[currVert]*3+3]
                    tempPixelData = [currTriangleVertices[currVert], vertexPixelValue[vPixels],vertexPixelValue[vPixels+1],currVert, currVertColour]
                    insideVertexDetailsToPPL.append(tempPixelData)
                    currVertexPixelData.append(tempPixelData)                            
                    if fflag == 0:
                        numberOfFullyInsideVertices+=1
                        fflag = 1
                    # currImage.append([vertexPixelValue[vPixels],vertexPixelValue[vPixels+1]])
                    # print("edge_v0: ", edge_v0," pixelvalue : ", vertexPixelValue)
                    # print("Adding data to PPL insideVertexDetailsToPPL")
            if fflag == 1:
                globalInsideVertexDataToPPL.append(currVertexPixelData)
                fullyInsideVerticesNumber.append(currVert)
        

            

        num_of_planes = 6
            
            
        # print("tr_vertex_outcodes = ", tr_vertex_outcodes)
        
        # for currPlane in range(0, num_of_planes):
        for currPlane in [5,0,1,2,3]:
            
            # print("\n\nCurrent plane = ", currPlane)
            tr_outside_vertex_set = []
            num_of_outside_vertices = 0
            
            for currVert in tr_vertices_set:
                if(tr_vertex_outcodes[currVert][currPlane] == 1):
                    
                    num_of_outside_vertices +=1
                    tr_outside_vertex_set.append(currVert)
            
            
            
            while(num_of_outside_vertices > 2):
                
                for currOutsideVert in tr_outside_vertex_set:               
                    edge1, edge2 = findEdges(currOutsideVert)
                    
                    if(tr_outside_vertex_set.count(edge1) and tr_outside_vertex_set.count(edge2)):
                        
                        removeEdges(currOutsideVert)
                        tr_vertices_set.remove(currOutsideVert)
                        tr_outside_vertex_set.remove(currOutsideVert)
                        tr_num_of_vertices = tr_num_of_vertices-1
                        edges.append([edge1,edge2])
                        break
                num_of_outside_vertices -=1
            
            
            if num_of_outside_vertices == 1:
                
                
                outsideVertex = tr_outside_vertex_set[0]
                
                edge1,edge2 = findEdges(outsideVertex)
                

                insideVertex1 = tr_vertex_coordinates[edge1]
                insideVertex1W = tr_vertex_ws[edge1]
                insideVertex2 = tr_vertex_coordinates[edge2]
                insideVertex2W = tr_vertex_ws[edge2]
                outsideVertex1 = tr_vertex_coordinates[outsideVertex]
                outsideVertex1W =tr_vertex_ws[outsideVertex]
                
                
                prop_t1, intersectionPoint1, intersectionPoint1W = cpp_vertexPlaneIntersectionPoint(insideVertex1, outsideVertex1, 
                                                                                    insideVertex1W, outsideVertex1W, currPlane)


                prop_t2, intersectionPoint2, intersectionPoint2W = cpp_vertexPlaneIntersectionPoint(insideVertex2, outsideVertex1,
                                                                                    insideVertex2W,outsideVertex1W, currPlane)
                
                
                
                if currPlane == 5:
                    intersectionPoint1W = -intersectionPoint1W
                    intersectionPoint2W = -intersectionPoint2W
            
                # print("pixel values of intersection points After \n---------")
                # print("intersectionPoint1 = ", pixelValue(intersectionPoint1, intersectionPoint1W))
                # print("intersectionPoint2 = ", pixelValue(intersectionPoint2, intersectionPoint2W))
                
                removeEdges(outsideVertex)
                tr_vertices_set.remove(outsideVertex)
                tr_num_of_vertices = tr_num_of_vertices-1
                
                
                tr_vertex_coordinates[tr_curr_num_of_vertex] = intersectionPoint1
                tr_vertex_ws[tr_curr_num_of_vertex] = intersectionPoint1W
                tr_vertices_set.append(tr_curr_num_of_vertex)                             
                outcode_t1 = clipCoordinateToOutcode(intersectionPoint1, intersectionPoint1W)
                tr_vertex_outcodes[tr_curr_num_of_vertex] = outcode_t1
                tr_num_of_vertices = tr_num_of_vertices+1  
                
                tempIntersectingData = [tr_curr_num_of_vertex, edge1, outsideVertex, currPlane]
                intersectingData[tr_curr_num_of_vertex] = tempIntersectingData
                
                redIntvert = currImageColours[edge1][0] + prop_t1 * (currImageColours[edge1][0] - currImageColours[outsideVertex][0])
                greenIntvert = currImageColours[edge1][1] + prop_t1 * (currImageColours[edge1][1] - currImageColours[outsideVertex][1])
                blueIntvert = currImageColours[edge1][2] + prop_t1 * (currImageColours[edge1][2] - currImageColours[outsideVertex][2])
                                            
                currImageColours[tr_curr_num_of_vertex] = [redIntvert, greenIntvert, blueIntvert]
                    
                tr_curr_num_of_vertex += 1
                
                
                
                
                tr_vertex_coordinates[tr_curr_num_of_vertex] = intersectionPoint2
                tr_vertex_ws[tr_curr_num_of_vertex] = intersectionPoint2W
                tr_vertices_set.append(tr_curr_num_of_vertex)                    
                outcode_t2 = clipCoordinateToOutcode(intersectionPoint2, intersectionPoint2W)
                tr_vertex_outcodes[tr_curr_num_of_vertex] =  outcode_t2
                tr_num_of_vertices = tr_num_of_vertices+1
                
                tempIntersectingData = [tr_curr_num_of_vertex, edge2, outsideVertex, currPlane]
                intersectingData[tr_curr_num_of_vertex] = tempIntersectingData
                
                
                redIntvert = currImageColours[edge2][0] + prop_t2 * (currImageColours[edge2][0] - currImageColours[outsideVertex][0])
                greenIntvert = currImageColours[edge2][1] + prop_t2 * (currImageColours[edge2][1] - currImageColours[outsideVertex][1])
                blueIntvert = currImageColours[edge2][2] + prop_t2 * (currImageColours[edge2][2] - currImageColours[outsideVertex][2])
                                            
                currImageColours[tr_curr_num_of_vertex] = [redIntvert, greenIntvert, blueIntvert]
                
                
                tr_curr_num_of_vertex += 1
                
                edges.append([tr_curr_num_of_vertex-2,edge1])
                edges.append([tr_curr_num_of_vertex-1,edge2])
                edges.append([tr_curr_num_of_vertex-1,tr_curr_num_of_vertex-2])


            elif num_of_outside_vertices == 2:
                
                
                outsideVertex1 = tr_outside_vertex_set[0]
                outsideVertex2 = tr_outside_vertex_set[1]
                
                edge1_1, edge1_2 =  findEdges(outsideVertex1)

                insidevertexOfoutside1 = 0
                outsidevertexOfOutside1 = 0
                
                if(edge1_1 != outsideVertex2):
                    insidevertexOfoutside1 = edge1_1
                else:
                    insidevertexOfoutside1 = edge1_2
                
                if(edge1_1 != insidevertexOfoutside1):
                    outsidevertexOfOutside1 = edge1_1
                else:
                    outsidevertexOfOutside1 = edge1_2
                
                
                
                insideVertex1_cord = tr_vertex_coordinates[insidevertexOfoutside1]
                insideVertex1W = tr_vertex_ws[insidevertexOfoutside1]                    
                outsideVertex1_cord = tr_vertex_coordinates[outsideVertex1]
                outsideVertex1W =tr_vertex_ws[outsideVertex1]
                
                prop_t1, intersectionPoint1, intersectionPoint1W = cpp_vertexPlaneIntersectionPoint(insideVertex1_cord, outsideVertex1_cord, 
                                                                                    insideVertex1W, outsideVertex1W, currPlane)

                removeEdges(outsideVertex1)
        
                tr_vertices_set.remove(outsideVertex1)
                tr_num_of_vertices = tr_num_of_vertices-1
                
                
                
                if currPlane == 5:
                    intersectionPoint1W = -intersectionPoint1W
                    
                
                tr_vertex_coordinates[tr_curr_num_of_vertex] = intersectionPoint1
                tr_vertex_ws[tr_curr_num_of_vertex] = intersectionPoint1W
                tr_vertices_set.append(tr_curr_num_of_vertex)
                tr_num_of_vertices = tr_num_of_vertices+1
                
                outcode_t1 = clipCoordinateToOutcode(intersectionPoint1,  intersectionPoint1W)
                tr_vertex_outcodes[tr_curr_num_of_vertex] =  outcode_t1
                
                tempIntersectingData = [tr_curr_num_of_vertex, insidevertexOfoutside1, outsideVertex1, currPlane]
                intersectingData[tr_curr_num_of_vertex] = tempIntersectingData
                
                
                redIntvert = currImageColours[insidevertexOfoutside1][0] + prop_t1 * (currImageColours[insidevertexOfoutside1][0] - currImageColours[outsideVertex1][0])
                greenIntvert = currImageColours[insidevertexOfoutside1][1] + prop_t1 * (currImageColours[insidevertexOfoutside1][1] - currImageColours[outsideVertex1][1])
                blueIntvert = currImageColours[insidevertexOfoutside1][2] + prop_t1 * (currImageColours[insidevertexOfoutside1][2] - currImageColours[outsideVertex1][2])
                                            
                currImageColours[tr_curr_num_of_vertex] = [redIntvert, greenIntvert, blueIntvert]
                
                
                tr_curr_num_of_vertex += 1
                
                edges.append([tr_curr_num_of_vertex-1,insidevertexOfoutside1])
                edges.append([tr_curr_num_of_vertex-1,outsidevertexOfOutside1])

                
                
                edge2_1, edge2_2 =  findEdges(outsideVertex2)
                
                
                insidevertexOfoutside2 = 0
                outsidevertexOfOutside2 = 0
                
                if(edge2_1 != outsideVertex1):
                    insidevertexOfoutside2 = edge2_1
                else:
                    insidevertexOfoutside2 = edge2_2
                    
                if(edge2_1 != insidevertexOfoutside2):
                    outsidevertexOfOutside2 = edge2_1
                else:
                    outsidevertexOfOutside2 = edge2_2
                
                
                insideVertex2_cord = tr_vertex_coordinates[insidevertexOfoutside2]
                insideVertex2W = tr_vertex_ws[insidevertexOfoutside2]
                
                outsideVertex2_cord = tr_vertex_coordinates[outsideVertex2]
                outsideVertex2W = tr_vertex_ws[outsideVertex2]
                
                prop_t2, intersectionPoint2, intersectionPoint2W = cpp_vertexPlaneIntersectionPoint(insideVertex2_cord, outsideVertex2_cord,
                                                                                                    insideVertex2W,outsideVertex2W, currPlane)
                
                removeEdges(outsideVertex2)      
                tr_vertices_set.remove(outsideVertex2)
                tr_num_of_vertices = tr_num_of_vertices-1
                
                if currPlane == 5:
                    intersectionPoint2W = -intersectionPoint2W
                
                tr_vertex_coordinates[tr_curr_num_of_vertex] = intersectionPoint2
                tr_vertex_ws[tr_curr_num_of_vertex] = intersectionPoint2W
                tr_vertices_set.append(tr_curr_num_of_vertex)
                tr_num_of_vertices = tr_num_of_vertices+1
                outcode_t1 = clipCoordinateToOutcode(intersectionPoint2,  intersectionPoint2W)
                tr_vertex_outcodes[tr_curr_num_of_vertex] =  outcode_t1
                
                tempIntersectingData = [tr_curr_num_of_vertex, insidevertexOfoutside2, outsideVertex2, currPlane]
                intersectingData[tr_curr_num_of_vertex] = tempIntersectingData
                
                
                redIntvert = currImageColours[insidevertexOfoutside2][0] + prop_t2 * (currImageColours[insidevertexOfoutside2][0] - currImageColours[outsideVertex2][0])
                greenIntvert = currImageColours[insidevertexOfoutside2][1] + prop_t2 * (currImageColours[insidevertexOfoutside2][1] - currImageColours[outsideVertex2][1])
                blueIntvert = currImageColours[insidevertexOfoutside2][2] + prop_t2 * (currImageColours[insidevertexOfoutside2][2] - currImageColours[outsideVertex2][2])
                                            
                currImageColours[tr_curr_num_of_vertex] = [redIntvert, greenIntvert, blueIntvert]
                
                
                tr_curr_num_of_vertex += 1

                edges.append([tr_curr_num_of_vertex-1,insidevertexOfoutside2])
                edges.append([tr_curr_num_of_vertex-1,outsidevertexOfOutside2])  
                

    if tr_num_of_vertices < 3:
        return 0, And(True)   


    intersectingVertices = list(set(tr_vertices_set) - set(fullyInsideVerticesNumber))
    # print("intersectingVertices: ", intersectingVertices)      
        
    set_option(rational_to_decimal=False)

   
    for currVert in intersectingVertices:
        
        currIntersectingData = intersectingData[currVert]
        
        
        currInsideVertCoordinates =[0,0,0]
        currOutsideVertCoordinates =[0,0,0]
        
        if currIntersectingData[1] <3:
            currInsideVertCoordinates = vertices[currTriangleVertices[currIntersectingData[1]]*3:currTriangleVertices[currIntersectingData[1]]*3+3]
        else:   
            tempCoordinates = outValueToWorldCoordinates(tr_vertex_coordinates[currIntersectingData[1]])
            currInsideVertCoordinates = [tempCoordinates[0]+posXp, tempCoordinates[1]+posYp, tempCoordinates[2]+posZp]
        
        if currIntersectingData[2] <3:
            currOutsideVertCoordinates = vertices[currTriangleVertices[currIntersectingData[2]]*3:currTriangleVertices[currIntersectingData[2]]*3+3]
        else:
            tempCoordinates = outValueToWorldCoordinates(tr_vertex_coordinates[currIntersectingData[2]])
            currOutsideVertCoordinates = [tempCoordinates[0]+posXp, tempCoordinates[1]+posYp, tempCoordinates[2]+posZp]
    
        currentIntersectingPlane = currIntersectingData[3]
        
        isIntersect, vertexPixelValue2,intersectionPoint,mp,mq =planeEdgeIntersectionUpdated(currentIntersectingPlane,currInsideVertCoordinates, currOutsideVertCoordinates,m,1)
        
        if( isIntersect== 1):            

            fflag =0
            currIntersectionData = []
            for vPixel in range(0, len(vertexPixelValue2),2):
                # print("vPixel = ", vPixel)
                if( (vertexPixelValue2[vPixel]>=0 and vertexPixelValue2[vPixel]<=49) and 
                    (vertexPixelValue2[vPixel+1]>=0 and vertexPixelValue2[vPixel+1]<=49)
                    ) : 
                    
                    if fflag == 0:
                        # numberOfIntersectingPlanes +=1
                        numberOfIntersectingEdges += 1
                        fflag = 1
                    
                    xpixel = vertexPixelValue2[vPixel]
                    ypixel = vertexPixelValue2[vPixel+1]    
                    
                    if((currIntersectingData[1] >2 and currIntersectingData[2] >2) and ((xpixel == 0 and ypixel ==0) or(xpixel ==48 and ypixel==0) or
                                                                                        (xpixel == 0 and ypixel ==48) or(xpixel ==48 and ypixel==48))):
                        
                        singleIntersectingData = [-2, currInsideVertCoordinates,currOutsideVertCoordinates,currentIntersectingPlane,\
                                            xpixel,ypixel, currIntersectingData[1], currIntersectingData[2], currIntersectingData[0]] 
                    else:                                                      
                        singleIntersectingData = [-1, currInsideVertCoordinates,currOutsideVertCoordinates,currentIntersectingPlane,\
                                            xpixel,ypixel, currIntersectingData[1], currIntersectingData[2], currIntersectingData[0]]                            
                    intersectingEdgeDataToPPL.append(singleIntersectingData) 
                    currIntersectionData.append(singleIntersectingData)  
                                                
                    # currImage.append([xpixel,ypixel])
            if fflag == 1:
                globalIntersectingVertexDataToPPL.append(currIntersectionData)
       
        
    currImageName = currGroupName+str(1)



    combinedDataToPPL = globalInsideVertexDataToPPL+ globalIntersectingVertexDataToPPL
    

    # print("computing pixel values: done ")
    # allImages.append(currImage)
    # print("Passing values to PPL for polyhedron computation") 
    
    
    
    
    
    
    fA =0
    tempRegionCons = []
    tempConsString = ""
    dataToComputeDepth = []
    
    howmanyData =0
    for k in itertools.product(*combinedDataToPPL):
        howmanyData +=1

        # print("\n",howmanyData," = ", k)
    
    currImageSetCons = And(True)

    currDataSet = 0
    for l in itertools.product(*combinedDataToPPL):

        environment.printLog(l)
        
        dataToComputeDepth = l
        
        currDataSet +=1

        insideVertexDetailsToPPL = l[0:numberOfFullyInsideVertices]
        intersectingEdgeDataToPPL = l[numberOfFullyInsideVertices:]

        
        currImageSetConStringPolyhedra = pyparma_posInvRegion40.computeRegion(currGroupName,posZp,numberOfFullyInsideVertices,insideVertexDetailsToPPL,\
        numberOfIntersectingEdges,intersectingEdgeDataToPPL,posXp,posYp,posZp,m[xp0],m[yp0],m[zp0], outcodeP0,currImageName)
    
        
        # print("inv region generated")
        currImageSetConString = str(currImageSetConStringPolyhedra.minimized_constraints())
        currImageSetConString = currImageSetConString.replace("x0","xp0")
        currImageSetConString = currImageSetConString.replace("x1","yp0")
        currImageSetConString = currImageSetConString.replace("x2","zp0")
        currImageSetConString = currImageSetConString.replace(" = ","==")
        currImageSetConString = currImageSetConString.replace("Constraint_System {"," ")
        currImageSetConString = currImageSetConString.replace("}"," ")
        
        if(str(currImageSetConString).replace(" ","") == "-1==0" or str(currImageSetConString).replace(" ","") == "0==-1"):
            
            # sleep(1)
            continue
        
        
        
        
        tempConsString = currImageSetConString
        # fA = 1
        currImageSetConString = "And("+str(currImageSetConString)+")"
        currImageSetCons = eval(currImageSetConString)
        # s2.add(Not(currImageSetCons))
        # print("Fully inside data")
        # print(l[0:numberOfFullyInsideVertices])
        # print("\n\n intersecting data")
        # print(l[numberOfFullyInsideVertices:])
        # if howmanyData >1 and currDataSet < howmanyData:             
        #     print("checking pos inclusion")
        scheck2 = Solver()
        scheck2.add(currImageSetCons)
        # scheck2.add(currGroupRegionCons)
        # # scheck.add(z3invRegionCons)
        
        # scheck.push()
        scheck2.add(And(xp0 ==m[xp0], yp0 == m[yp0], zp0 == m[zp0]))
        if(scheck2.check() != sat):                
            # sleep(3)
            continue
    
            
        
        # print("Exiting")
        
        # print(currImageSetCons)
        # print(simplify(tempRegionCons2))
        
        
        
        # s4 = Solver()
        # s4.add(tempRegionCons2 == currImageSetCons )
        
        # print(s4.check())
        
        # exit()
        
        
        fA = 1
        
        # s2.add(Not(currImageSetCons))             
        # tempRegionCons.append(currImageSetConString)             
        break    
        
    
    if fA == 0:
        # break
        # print("Current Triangle,  ", currTriangle)
        # print("numberOfInvRegions = ", numberOfInvRegions)
        # print("Fa failes")
        # print("howmanyData = ", howmanyData)

        foundR, consOfReg,minmaxDepths, centerPointImage = oldInvComputation1.computeInv( posXp, posYp, posZp, currTriangle,m)
        
        # print(foundR, consOfReg)
        scheck3 = Solver()
        # print(consOfReg)
        # print(eval("And("+str(consOfReg)+")"))
        scheck3.add(eval("And("+str(consOfReg)+")"))
        # scheck3.add(currGroupRegionCons)
        # scheck.add(z3invRegionCons)            
        
        # sleep(5)
        scheck3.add(And(xp0 ==m[xp0], yp0 == m[yp0], zp0 == m[zp0]))

        if(scheck3.check() == sat and foundR == 1 ):
            currImageSetCons = eval("And("+str(consOfReg)+")")
            # s2.add(Not(currImageSetCons))  
            # numberOfInvRegions +=1                
            # intervalImageFunctions1.updateSingleIntervalImage(currTriangleIntervalImage,  currTriangle, minmaxDepths, centerPointImage)
            
            # s2.add(notTheCurrentPosCons1)
            fA = 1

        else:                
            # # sleep(3)
            # print("Exiting.....")
            # print(m)
            # print(posXp, posYp, posZp)
            # print(eval("And("+str(consOfReg)+")"))
            # print("Breaking nm bound failed")
            # sleep(5)

            # regCons, currVarsIndex2 = oldInvRegionSolver.invUsingConstraints(posXp, posYp, posZp, currTriangle,m,
            #                                        currVarsIndex, pVars, aVars, bVars,cVars,dVars)


            # currVarsIndex = currVarsIndex2

            # s2.add(Not(regCons))

            # numberOfInvRegions +=1
            eFile = open("ErrorLog.txt","a")
            eFile.write("Error in finding the region for the triangle "+str(currTriangle)+"\n")
            eFile.write("Current position = "+str(posXp)+", "+str(posYp)+", "+str(posZp)+"\n")
            eFile.write("Current consOfReg = "+str(consOfReg)+"\n")
            eFile.close()

            nmBound = 0.0001
            # consOfReg2 =   str(m[xp0]-nmBound) +"<=xp0, xp0<="+ str(m[xp0]+nmBound)+"," \
            #     + str(m[yp0]-nmBound) +"<=yp0, yp0<="+ str(m[yp0]+nmBound)+"," \
            #     + str(m[zp0]-nmBound) +"<=zp0, zp0<="+ str(m[zp0]+nmBound)
            
            consOfReg2 =   str(posXp-nmBound) +"<=xp0, xp0<="+ str(posXp+nmBound)+"," \
                + str(posYp-nmBound) +"<=yp0, yp0<="+ str(posYp+nmBound)+"," \
                + str(posZp-nmBound) +"<=zp0, zp0<="+ str(posZp+nmBound)
            
            currImageSetCons = eval("And("+str(consOfReg2)+")")
            # print(eval("And("+str(consOfReg2)+")"))


            # print("Current triangle = ", currTriangle)
            # print("numberOfInvRegions = ", numberOfInvRegions)

            # print("currGroupRegionCons = ", currGroupRegionCons)
            # s2.add(Not(currImageSetCons))  
            # intervalImageFunctions1.updateSingleIntervalImage(currTriangleIntervalImage,  currTriangle, minmaxDepths, centerPointImage)
            

            # print("\n\n")

            # sleep(5) 
            # break
        

        
    
        
       



                    
        
        
        

            
    currImageSetConString = currImageSetCons

    return 1, currImageSetConString






def computeInvRegions(currGroupName, currGroupRegionCons, dnnOutput, fromSplitRegion=0):
    print("Invaraint Region Computation Started")
    print("currGroupName: ", currGroupName)
    print("currGroupRegionCons: ", currGroupRegionCons)


    regionsToReturn = []
    foundARegion = 0

    s2 = Solver()
    set_param('parallel.enable', True)
    set_option(rational_to_decimal=True)
    # set_option(precision=1000)
    set_param('parallel.enable', True)
    s2.set("sat.local_search_threads", 28)
    s2.set("sat.threads", 28)
    # s2.set("timeout",2000)    
    set_option(max_args=10000000, max_lines=1000000, max_depth=10000000, max_visited=1000000)

    s2.add(simplify(currGroupRegionCons))

    triaglesInvRegions = dict()
    triaglesInvRegions.clear()

    os.remove("ErrorLog.txt")
    with open("ErrorLog.txt", "w") as f:
        f.write(str(datetime.now())+"\n\n")

    numberOfInvRegions = 0
    while(s2.check() ==sat):
        numberOfInvRegions +=1
        print("Number of Invariant Regions: ", numberOfInvRegions)
        m = s2.model()
        print(m)

        sleep(2)
        
        posXp = (eval("m[xp0].numerator_as_long()/m[xp0].denominator_as_long()"))
        posYp = (eval("m[yp0].numerator_as_long()/m[yp0].denominator_as_long()"))
        posZp = (eval("m[zp0].numerator_as_long()/m[zp0].denominator_as_long()"))

        notTheCurrentPosCons1 = Or(xp0!= m[xp0], yp0!=m[yp0],zp0!= m[zp0])
        for i in range(0, scene.numOfTriangles):
            regionExist, currTriangleInvRegion = getInvariantRegion(i, posXp, posYp, posZp, m,currGroupName)

            # print("currTriangleInvRegion: ", currTriangleInvRegion)
            
            if regionExist == 1:    
                triaglesInvRegions[i] = currTriangleInvRegion
        

        # if len(triaglesInvRegions) > 0:

        currInvCons = And(True)
        for currInv in triaglesInvRegions.values():
            # print("Invariant Region:: ", currInv)
           
            # print("\n\n")
            # # if triaglesInvRegions[currInv] == 0:
            #     continue
            currInvCons = And(currInvCons, currInv)
        


        pythonRenderAnImage2.renderAnImage(posXp,posYp, posZp,"RefineImage2_0")
    
        
        print("running dnn")
        print("images/"+str("RefineImage2_0"))
        # currImageDnnOutput= getDNNOutput("images/"+str("RefineImage2_0.ppm"))
        # currImageDnnOutput = 1
        print("running on second model")
        iisc_net_dnnoutput = getDNNOutput_onnx("images/"+str("RefineImage2_0.ppm"),'OGmodel_pb_converted.onnx')
        
        if iisc_net_dnnoutput == dnnOutput:
            regionsToReturn.append([currInvCons])
            foundARegion = 1
        
        print("dnnOutput: ", dnnOutput)
        print("iisc_net_dnnoutput: ", iisc_net_dnnoutput)
        print("Number of Invariant Regions: ", numberOfInvRegions)
        sleep(5)
        
        s2.add(Not(currInvCons))
        
        s2.add(notTheCurrentPosCons1)

        

    print("Invariant Region Computation Done")
    print("Number of Invariant Regions: ", numberOfInvRegions)
    return foundARegion, regionsToReturn


       








# currGroupName ="G_"
# # currGroupRegionCons = environment.initCubeCon

# currGroupRegionCons = And(10*xp0>=36,1000*xp0<=3605,10*yp0>=45,10*yp0<=45, 10*zp0>=1755,10*zp0<=1755)

# computeInvRegions(currGroupName, currGroupRegionCons)












































