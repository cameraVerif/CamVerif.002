from pyparma import *

import environment
import math
from time import sleep
import numpy as np


import matplotlib.pyplot as plt
from collections import Counter


import math
import time
from time import time
from datetime import datetime 
import anytree
import os
import sys
import itertools

import singleTriangleInvRegions31_P3_1



vertices = environment.vertices
numOfVertices = environment.numOfVertices
tedges = environment.tedges
numOftedges = environment.numOfEdges
nvertices = environment.nvertices
vertColours = environment.vertColours

imageWidth = environment.imageWidth
imageHeight = environment.imageHeight


canvasWidth = environment.canvasWidth
canvasHeight = environment.canvasHeight
focalLength = environment.focalLength
t= environment.t
b = environment.b
l = environment.l
r = environment.r
n = environment.n
f = environment.f


singleTriangleIntervalImageData = dict()
globalIntervalImageP3 = dict()


# OpenGL perspective projection matrix
mProj = [
        [2 * n / (r - l), 0, 0, 0],
        [0,2 * n / (t - b),0,0],
        [(r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n), -1 ],
        [0,0,-2 * f * n / (f - n),0]
    ]
def computeOutcodeAtPos2( inx, iny,inz):
    
    
    outx   = inx * mProj[0][0] + iny * mProj[1][0] + inz * mProj[2][0] +  mProj[3][0]
    outy   = inx * mProj[0][1] + iny * mProj[1][1] + inz * mProj[2][1] +  mProj[3][1] 
    outz   = inx * mProj[0][2] + iny * mProj[1][2] + inz * mProj[2][2] +  mProj[3][2] 
    w      = inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3] 
    
    outValueToReturn = [outx, outy, outz]
  
        
    return outValueToReturn, w


def pixelValue(point, w):
    t0 = point[0]/w
    t1 = point[1]/w
    t2 = point[2]/w
    
    # print(((t0 + 1) * 0.5 * imageWidth),((1 - (t1 + 1) * 0.5) * imageHeight), t2)
    originalPixel = [((t0 + 1) * 0.5 * imageWidth),(( (t1 + 1) * 0.5) * imageHeight), t2]
    # print("pixel value before min = ",originalPixel  )
    
    raster0 = min(imageWidth - 1, int((t0 + 1) * 0.5 * imageWidth))
    raster1 = min(imageHeight - 1, int((1 - (t1 + 1) * 0.5) * imageHeight))
    raster2 = t2
    
    
    
    return  originalPixel, [raster0, raster1, raster2]
def edgeFunction(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])     
    



def prepareGlobalIntervalImageP3():
    
    #for each triangle present in the singleTriangle Interval image
    #for each pixel
    #if the pixel present in the global interval image
    #if the current pixel depth interval is less than the global depth then replace color intervals
    #if it overlaps then expand both the depht and colour intervals   
    
    for ct in singleTriangleIntervalImageData:
        currTriangleData = singleTriangleIntervalImageData[ct]
        
        # currTriangleColour = [ vertColours[nvertices[ct*3]], vertColours[nvertices[ct*3 + 1]],
        #                       vertColours[nvertices[ct*3+2]]]
        currTriangleColour = [ vertColours[nvertices[ct*3]+0], vertColours[nvertices[ct*3] + 1],
                              vertColours[nvertices[ct*3]+2]]
        

        print("currTriangleData = ",currTriangleData)
        # sleep(0.5)
        currTriangleMinDepth = currTriangleData[0]
        currTriangleMaxDepth = currTriangleData[1]
        for currPixel in currTriangleData[2]:
            print(currPixel, end=" ")
            pixelIndex = currPixel[1]*imageWidth+currPixel[0]
            newDataToStore = [currTriangleMinDepth, 1000,
                             currTriangleColour[0]*255, currTriangleColour[0]*255,
                             currTriangleColour[1]*255, currTriangleColour[1]*255,
                             currTriangleColour[2]*255, currTriangleColour[2]*255]
            
            if globalIntervalImageP3.get(pixelIndex):
                print(" Data Already there")
                currGlobalData = globalIntervalImageP3[pixelIndex]
                # currMinD  = currGlobalData[0]
                currMaxD  = currGlobalData[1]
                
                # if currTriangleMaxDepth <= currMinD:
                #     #replace the colours
                #     globalIntervalImageP3[pixelIndex] = newDataToStore
                # elif currTriangleMaxDepth > currMaxD:
                #     pass
                # else:
                if currTriangleMinDepth < currMaxD:
                    globalIntervalImageP3[pixelIndex] = [
                        min(currGlobalData[0],newDataToStore[0]),
                        max(currGlobalData[1],newDataToStore[1]),                        
                        min(currGlobalData[2],newDataToStore[2],1),
                        max(currGlobalData[3],newDataToStore[3],1),
                        min(currGlobalData[4],newDataToStore[4],25),
                        max(currGlobalData[5],newDataToStore[5],25),
                        min(currGlobalData[6],newDataToStore[6],24),
                        max(currGlobalData[7],newDataToStore[7],24)  
                    ]
            
            else:
                print("New Data")
                globalIntervalImageP3[pixelIndex] = newDataToStore
                
        for currPixel in currTriangleData[3]:
            print(currPixel, end=" ")
            pixelIndex = currPixel[1]*imageWidth+currPixel[0]
            newDataToStore = [currTriangleMinDepth, currTriangleMaxDepth,
                             currTriangleColour[0], currTriangleColour[0],
                             currTriangleColour[1], currTriangleColour[1],
                             currTriangleColour[2], currTriangleColour[2]]
            
            if globalIntervalImageP3.get(pixelIndex):
                print(" Data Already there")
                currGlobalData = globalIntervalImageP3[pixelIndex]
                currMinD  = currGlobalData[0]
                currMaxD  = currGlobalData[1]
                
                if currTriangleMaxDepth <= currMinD:
                    #replace the colours
                    globalIntervalImageP3[pixelIndex] = newDataToStore
                elif currTriangleMinDepth > currMaxD:
                    pass
                else:
                # if currTriangleMinDepth < currMaxD:
                    globalIntervalImageP3[pixelIndex] = [
                        min(currGlobalData[0],newDataToStore[0]),
                        max(currGlobalData[1],newDataToStore[1]),                        
                        min(currGlobalData[2],newDataToStore[2]),
                        max(currGlobalData[3],newDataToStore[3]),
                        min(currGlobalData[4],newDataToStore[4]),
                        max(currGlobalData[5],newDataToStore[5]),
                        min(currGlobalData[6],newDataToStore[6]),
                        max(currGlobalData[7],newDataToStore[7])  
                    ]
            
            else:
                print("New Data")
                globalIntervalImageP3[pixelIndex] = newDataToStore
        


def updateGlobalIntervalImageForIntersectingTriangles(intersectingTrianglesIntImage):

    print(globalIntervalImageP3)

    for ct in intersectingTrianglesIntImage:
        currTriangleData  = intersectingTrianglesIntImage[ct]
        # print(currTriangleData)
        # print(len(currTriangleData))
        

        # currTriangleColour = [ vertColours[nvertices[ct*3]], vertColours[nvertices[ct*3 + 1]],
        #                       vertColours[nvertices[ct*3+2]]]
       
        for currPixel in currTriangleData:

            currPixelData = currTriangleData[currPixel]

            if globalIntervalImageP3.get(currPixel):
                print(" Data Already there")
                currGlobalData = globalIntervalImageP3[currPixel]
                currMinD  = currGlobalData[0]
                currMaxD  = currGlobalData[1]

                currTriangleMaxDepth = currPixelData[7]
                currTriangleMinDepth = currPixelData[6]
                if currTriangleMaxDepth <= currMinD:
                    newDataToStore = [currPixelData[6], currPixelData[7],
                             currPixelData[0], currPixelData[1],
                             currPixelData[2], currPixelData[3],
                             currPixelData[4], currPixelData[5]]
                    globalIntervalImageP3[currPixel] = newDataToStore

                elif currTriangleMinDepth > currMaxD:
                    
                    pass

                else:
                    # print("depth intesect")
                    globalIntervalImageP3[currPixel] = [
                        min(currGlobalData[0],currPixelData[6]),
                        max(currGlobalData[1],currPixelData[7]),                        
                        min(currGlobalData[2],currPixelData[0]),
                        max(currGlobalData[3],currPixelData[1]),
                        min(currGlobalData[4],currPixelData[2]),
                        max(currGlobalData[5],currPixelData[3]),
                        min(currGlobalData[6],currPixelData[4]),
                        max(currGlobalData[7],currPixelData[5])  
                    ]

            else:
                # print(" New Data")
                # print("currPixelData :", currPixelData)
                newDataToStore = [currPixelData[6], currPixelData[7],
                             currPixelData[0], currPixelData[0],
                             currPixelData[1], currPixelData[1],
                             currPixelData[2], currPixelData[2]]
                globalIntervalImageP3[currPixel] = newDataToStore


            

            # currTriangleMinDepth = currTriangleData[currPixel][6]
            # currTriangleMaxDepth = currTriangleData[currPixel][7]

            # print(currPixel,": ", currTriangleMinDepth, currTriangleMaxDepth)

        




       
                    
        
        
    


def drawTriangle4(raster0,raster1,raster2):
    
    centerPoint = [0,0,0]
    
    centerPoint[0] = (raster0[0]+raster1[0]+raster2[0])/3
    centerPoint[1] = (raster0[1]+raster1[1]+raster2[1])/3
    
    PI = 3.14159265358979323846
    
    angle0 = math.atan2(raster0[1] - centerPoint[1],  raster0[0] - centerPoint[0]) * 180 / PI
    angle1 = math.atan2(raster1[1] - centerPoint[1],  raster1[0] - centerPoint[0]) * 180 / PI
    angle2 = math.atan2(raster2[1] - centerPoint[1],  raster2[0] - centerPoint[0]) * 180 / PI
    
    angle0 = angle0 if angle0>0 else (int)(angle0 + 360) % 360
    angle1 = angle1 if angle1>0 else (int)(angle1 + 360) % 360
    angle2 = angle2 if angle2>0 else (int)(angle2 + 360) % 360
    
    minAngle = min(angle0, angle1, angle2)
    
    print("Center Point: " + str(centerPoint))
    print("Angle0: " + str(angle0))
    print("Angle1: " + str(angle1))
    print("Angle2: " + str(angle2))
    print("Min Angle: " + str(minAngle))
    
    
    tempV0 = [0,0,0]
    tempV1 = [0,0,0]
    tempV2 = [0,0,0]
    v0flag = 0
    v1flag = 0
    v0MinDepth = 0
    v0MaxDepth = 0
    v1MinDepth = 0
    v1MaxDepth = 0
    v2MinDepth = 0
    v2MaxDepth = 0
       
    
    if(minAngle == angle0):
        tempV0 = raster0
        v0flag =0
        # v0MinDepth = d0
        # v0MaxDepth = d1
    elif(minAngle == angle1):
        tempV0 = raster1
        v0flag =1
        # v0MinDepth = d2
        # v0MaxDepth = d3
    else:
        tempV0 = raster2
        v0flag =2
        # v0MinDepth = d4
        # v0MaxDepth = d5
    
 
    if(v0flag == 0):
        if(angle1<=angle2):
            tempV1 = raster1
            v1flag =1
            # v1MinDepth = d2
            # v1MaxDepth = d3
        else:
            tempV1 = raster2
            v1flag =2
            # v1MinDepth = d4
            # v1MaxDepth = d5
    elif(v0flag == 1):
        if(angle0<=angle2):
            tempV1 = raster0
            v1flag =0
            # v1MinDepth = d0
            # v1MaxDepth = d1
        else:
            tempV1 = raster2
            v1flag =2
            # v1MinDepth = d4
            # v1MaxDepth = d5
    else:
        if(angle0<=angle1):
            tempV1 = raster0
            v1flag =0
            # v1MinDepth = d0
            # v1MaxDepth = d1
        else:
            tempV1 = raster1
            v1flag =1
            # v1MinDepth = d2
            # v1MaxDepth = d3
    
    
    if(v0flag != 0 and v1flag != 0 ):
        tempV2 = raster0
        # v2MinDepth = d0
        # v2MaxDepth = d1
    elif(v0flag != 1 and v1flag != 1 ):
        tempV2 = raster1
        # v2MinDepth = d2
        # v2MaxDepth = d3
    else:
        tempV2 = raster2
        # v2MinDepth = d4
        # v2MaxDepth = d5
    
    
    v0Raster = [0,0,0]
    v1Raster = [0,0,0]
    v2Raster = [0,0,0]
    
    v0Raster[0] = tempV2[0]
    v0Raster[1] = tempV2[1]
    v0Raster[2] = tempV2[2]
    
    v1Raster[0] = tempV1[0]
    v1Raster[1] = tempV1[1]
    v1Raster[2] = tempV1[2]
    
    v2Raster[0] = tempV0[0]
    v2Raster[1] = tempV0[1]
    v2Raster[2] = tempV0[2]
    
    xmin = min(v0Raster[0], v1Raster[0], v2Raster[0])
    ymin = min(v0Raster[1], v1Raster[1], v2Raster[1])
    xmax = max(v0Raster[0], v1Raster[0], v2Raster[0])
    ymax = max(v0Raster[1], v1Raster[1], v2Raster[1])
    
    if (xmin > imageWidth - 1 or xmax < 0 or ymin > imageHeight - 1 or ymax < 0):
        print("Out of screen")
        return
    
    print("((((((((((Drawing Triangle)))))))))))))")
    print("v0Raster: " + str(v0Raster)+ " , v1Raster: " + str(v1Raster)+ " , v2Raster: " + str(v2Raster))
    
    
    x0 = max(0, (int)(math.floor(xmin)))
    x1 = min(imageWidth - 1, (int)(math.floor(xmax)))
    y0 = max(0, (int)(math.floor(ymin)))
    y1 = min(imageHeight - 1, (int)(math.floor(ymax)))

    print("x0: " + str(x0) + " , x1: " + str(x1) + " , y0: " + str(y0) + " , y1: " + str(y1))
    
    area = edgeFunction(v0Raster, v1Raster, v2Raster)
    
    print("area: " + str(area))

    if (area <= 0):
        return

    px = Variable(0)
    py = Variable(1)
    polyTemp2 = NNC_Polyhedron(2,'empty')
    polyTemp2.add_generator(point(v0Raster[0]*px+49-v0Raster[1]*py))
    polyTemp2.add_generator(point(v1Raster[0]*px+49-v1Raster[1]*py))
    polyTemp2.add_generator(point(v2Raster[0]*px+49-v2Raster[1]*py))
    print(polyTemp2.constraints())


    pixelToColour = []
    # pixelToColour2 = []
    for y in range(y0, y1+1):
        for x in range(x0, x1+1):
            pixelSample = [x + 0.5, y + 0.5, 0]
            w0 = edgeFunction(v1Raster, v2Raster, pixelSample)
            w1 = edgeFunction(v2Raster, v0Raster, pixelSample)
            w2 = edgeFunction(v0Raster, v1Raster, pixelSample)


            # centerX = str(x) + "5"
            # centerY = str(y) + "5"
            # # print(x, y , centerX, centerY)

            # pdTemp = NNC_Polyhedron(2,'empty')
            # pdTemp.add_generator(point(int(centerX)*px+int(centerY)*py, pow(10,1)))

            # if(polyTemp2.contains(pdTemp)):  
            #     pixelToColour2.append([x, y]) 

           
            if (w0 >= 0 and w1 >= 0 and w2 >= 0):
                w0 = w0 / area
                w1 = w1 / area
                w2 = w2 / area
                oneOverZ = v0Raster[2] * w0 + v1Raster[2] * w1 + v2Raster[2] * w2
                z = 1 / oneOverZ
                storeZasDepth = z

                print( "(",x,y,")", end=" ")

                pixelToColour.append([x, y])
                
                
                
                # r = w0 * currVertexColours[0][0]*255 + w1 * currVertexColours[1][0]*255 + w2 * currVertexColours[2][0]*255 
                # g = w0 * currVertexColours[0][1]*255 + w1 * currVertexColours[1][1]*255 + w2 * currVertexColours[2][1]*255
                # b = w0 * currVertexColours[0][2]*255 + w1 * currVertexColours[1][2]*255 + w2 * currVertexColours[2][2]*255
                
                # currMinDepth = v0MinDepth * w0 + v1MinDepth * w1 + v2MinDepth * w2
                # currMaxDepth = v0MaxDepth * w0 + v1MaxDepth * w1 + v2MaxDepth * w2
               
                # frameBuffer[y * imageWidth + x] = [int(r),int(g),int(b)]
                
                # if currTriangleIntervalImage.get(y*imageWidth+x):
                #     # print(y*imageWidth+x, "already exists")
                #     currValues = currTriangleIntervalImage[y*imageWidth+x]
                #     currValues.append([r,g,b,currMinDepth,currMaxDepth])
                #     currTriangleIntervalImage[y*imageWidth+x] = currValues
                # else:
                #     currTriangleIntervalImage[y*imageWidth+x] = [[r,g,b,currMinDepth,currMaxDepth]]

                # # print(y*imageWidth + x,int(r),int(g),int(b))
    
    # print("pixelToColour2 = ",pixelToColour2)
    return pixelToColour
     

def computeConvexHull(points):
    px = Variable(0)
    py = Variable(1)
    
    pd =NNC_Polyhedron(2,'empty')
    
    for p in points:        
        pd.add_generator(point(p[0]*px+p[1]*py, pow(10,4)))
    
    print(pd.constraints())
    
    gs = pd.generators()# // Use ph.minimized_generators() to minimal set of points for the polytope

    print(gs)

    vertString = str(gs)
    vertString = vertString.replace("Generator_System {","").replace("}","").replace("point","").replace("(","").replace(")","")
    cornerPoints = vertString.split(",")
    print(cornerPoints)  

    return pd.constraints(), cornerPoints

def computeConvexHull2(points):
    px = Variable(0)
    py = Variable(1)
    
    pd =NNC_Polyhedron(2,'empty')
    
    for p in points:        
        pd.add_generator(point(p[0]*px+p[1]*py))
    
    # print(pd.constraints())
    
    # gs = pd.generators()# // Use ph.minimized_generators() to minimal set of points for the polytope

    # print(gs)

    # vertString = str(gs)
    # vertString = vertString.replace("Generator_System {","").replace("}","").replace("point","").replace("(","").replace(")","")
    # cornerPoints = vertString.split(",")
    # print(cornerPoints)  

    # return pd, cornerPoints
    return pd

def computeConvexHull3(points):
    px = Variable(0)
    py = Variable(1)
    
    pd =NNC_Polyhedron(2,'empty')
    
    for p in points:        
        pd.add_generator(point(p[0]*px+p[1]*py, pow(10,4)))
    
    # print(pd.constraints())
    
    # gs = pd.generators()# // Use ph.minimized_generators() to minimal set of points for the polytope

    # print(gs)

    # vertString = str(gs)
    # vertString = vertString.replace("Generator_System {","").replace("}","").replace("point","").replace("(","").replace(")","")
    # cornerPoints = vertString.split(",")
    # print(cornerPoints)  

    # return pd, cornerPoints
    return pd

def getHullCornerPoints(hullCons):
    gs = hullCons.minimized_generators()# // Use ph.minimized_generators() to minimal set of points for the polytope

    print(gs)

    vertString = str(gs)
    vertString = vertString.replace("Generator_System {","").replace("}","").replace("point","").replace("(","").replace(")","")
    cornerPoints = vertString.split(",")
    print(cornerPoints)  

    return cornerPoints




def generate_vnnlib_files2(globalIntervalImageP3):
    
    
    tempString = ""

    for i in range(0,49*49*3):
        # print(f"(declare-const X_{i} Real)")
        tempString += "(declare-const X_"+str(i)+" Real)\n"

    tempString += "(declare-const Y_0 Real)\n"
    tempString += "(declare-const Y_1 Real)\n"
    tempString += "(declare-const Y_2 Real)\n\n\n"
    
    for i in range(0,49*49):
        # print(f"(assert (<= X_{i} 0.679857769))")   
        # print(f"(assert (>= X_{i} 0.268978427))\n") 
        
        if globalIntervalImageP3.get(i):
            # print(i*3," ==> ", globalIntervalImage[i])
                    
            tempString += "(assert (>= X_"+str(i*3+0)+" "+str(globalIntervalImageP3[i][2]/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+0)+" "+str(globalIntervalImageP3[i][3]/255)+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+1)+" "+str(globalIntervalImageP3[i][4]/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+1)+" "+str(globalIntervalImageP3[i][5]/255)+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+2)+" "+str(globalIntervalImageP3[i][6]/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+2)+" "+str(globalIntervalImageP3[i][7]/255)+"))\n"
        else:
            tempString += "(assert (>= X_"+str(i*3+0)+" "+str(1/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+0)+" "+str(1/255)+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+1)+" "+str(25/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+1)+" "+str(25/255)+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+2)+" "+str(24/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+2)+" "+str(24/255)+"))\n"
            
        
        
    
    f = open("prop_y0.vnnlb", "w")
    f.write(tempString)

    tempString2 = "(assert (or\n"
    tempString2 += " (and (>= Y_0 Y_1) (>= Y_0 Y_2))))"
    f.write(tempString2)
    f.close()   

    # print("(assert (or")
    # print(" (and (>= Y_0 Y_1) (>= Y_0 Y_2))))")   
        
    f = open("prop_y1.vnnlb", "w")
    f.write(tempString)
    tempString2 = "(assert (or\n"
    tempString2 += " (and (>= Y_1 Y_0) (>= Y_1 Y_2))))"
    f.write(tempString2)
    f.close()  

    f = open("prop_y2.vnnlb", "w")
    f.write(tempString)
    tempString2 = "(assert (or\n"
    tempString2 += " (and (>= Y_2 Y_0) (>= Y_2 Y_1))))"
    f.write(tempString2)
    f.close()  

    del tempString2
    del tempString
    

def computeIntervalImage_P3(currAbsGroupName, currAbsGroupRegionCons, currRegionMinMaxValues, 
                            currRegionCornerPoints,fromSplitRegion=0):   

    numberOfIntersectingTriangles = 0
    numberOfFullyBacksideTriangles = 0
    numberOfEmptyIntersectionTriangles = 0
    
    
    singleTriangleIntervalImageData.clear()  
    globalIntervalImageP3.clear()

    px = Variable(0)
    py = Variable(1)

    canvasPolyhedra = NNC_Polyhedron(2,'empty')
    canvasPolyhedra.add_generator(point(0*px+0*py))
    canvasPolyhedra.add_generator(point(49*px+0*py))
    canvasPolyhedra.add_generator(point(0*px+49*py))
    canvasPolyhedra.add_generator(point(49*px+49*py))


    print(canvasPolyhedra.constraints())

    print("currRegionMinMaxValues =  ", currRegionMinMaxValues)
    print("currRegionCornerPoints =  ", currRegionCornerPoints)
    
    # sleep(5)

    

    intersectingTriangles = []

    for currTriangle in range(0,environment.numOfTriangles):
    # for currTriangle in range(0,100):
        print("\n-----------------\ncurrernt Triangle::: ", currTriangle)

        vertex0 = nvertices[currTriangle*3+0]
        vertex1 = nvertices[currTriangle*3+1]
        vertex2 = nvertices[currTriangle*3+2]
        currTriangleVertices = [vertex0, vertex1,vertex2]

        v0Vertex = [vertices[currTriangleVertices[0]*3+0], vertices[currTriangleVertices[0]*3+1],vertices[currTriangleVertices[0]*3+2] ]
        v1Vertex = [vertices[currTriangleVertices[1]*3+0], vertices[currTriangleVertices[1]*3+1],vertices[currTriangleVertices[1]*3+2] ]
        v2Vertex = [vertices[currTriangleVertices[2]*3+0], vertices[currTriangleVertices[2]*3+1],vertices[currTriangleVertices[2]*3+2] ]

        print("Current Triangle Info")
        print("Triangle Id: ", currTriangle)
        print("Vertices index : ", currTriangleVertices)
        print("Vertices : ", v0Vertex, "\n", v1Vertex, "\n", v2Vertex)
        
        
        if Counter(v0Vertex) == Counter(v1Vertex) or Counter(v0Vertex) == Counter(v2Vertex) or Counter(v1Vertex) == Counter(v2Vertex):
            print(v0Vertex, v1Vertex, v2Vertex)
            print("Error triangle")
            print(currTriangle)
            continue
        
        # if(vertices[currTriangleVertices[0]*3+2] > posZp100+1 and vertices[currTriangleVertices[1]*3+2] > posZp100+1 and vertices[currTriangleVertices[2]*3+2] > posZp100+1 ):
        #     print("Invisible from region")
        #     return 0
        
        
        vertexProjectedPoints = dict()
        vertexProjectedPoints.clear()

        if( v0Vertex[2] <= currRegionMinMaxValues[4]-1  and v1Vertex[2] <= currRegionMinMaxValues[4]-1 and
            v2Vertex[2] <= currRegionMinMaxValues[4]-1):
            print("Triangle is fully in front of the region")

            pointsToComputeConvexHull = []
            
            v0ProjectionToComputeConvexHull =[]
            v1ProjectionToComputeConvexHull =[]
            v2ProjectionToComputeConvexHull =[]

            depthValues = []
            
            
            for currCornerPoint in currRegionCornerPoints:
                print("\nCurrent Corner Point: ", currCornerPoint)
                posXp = currCornerPoint[0]
                posYp = currCornerPoint[1]
                posZp = currCornerPoint[2]
            
                
                outValue0, outW0 = computeOutcodeAtPos2(
                                v0Vertex[0]-posXp ,
                                v0Vertex[1]-posYp,
                                v0Vertex[2]-posZp )
                outValue1, outW1 = computeOutcodeAtPos2(
                                v1Vertex[0]-posXp ,
                                v1Vertex[1]-posYp,
                                v1Vertex[2]-posZp )

                outValue2, outW2 = computeOutcodeAtPos2( 
                                v2Vertex[0]-posXp ,
                                v2Vertex[1]-posYp,
                                v2Vertex[2]-posZp)

                projection0, pixel0 = pixelValue(outValue0,outW0)
                projection1, pixel1  = pixelValue(outValue1,outW1)
                projection2, pixel2  = pixelValue(outValue2,outW2)

                print(pixel0, pixel1, pixel2)
                print(projection0, projection1, projection2)

                # originalPixelToColor = drawTriangle4(pixel0, pixel1, pixel2)


                projection0[0] = int(projection0[0]*10000)
                projection0[1] = int(projection0[1]*10000)
                projection1[0] = int(projection1[0]*10000)
                projection1[1] = int(projection1[1]*10000)
                projection2[0] = int(projection2[0]*10000)
                projection2[1] = int(projection2[1]*10000)
                
                v0ProjectionToComputeConvexHull.append(projection0)
                v1ProjectionToComputeConvexHull.append(projection1)
                v2ProjectionToComputeConvexHull.append(projection2)

                depthValues.append(projection0[2])
                depthValues.append(projection1[2])
                depthValues.append(projection2[2])
            
            
            print("\n\n")
            print("v0ProjectionToComputeConvexHull: ", v0ProjectionToComputeConvexHull)
            print("v1ProjectionToComputeConvexHull:",v1ProjectionToComputeConvexHull)
            print("v2ProjectionToComputeConvexHull: ",v2ProjectionToComputeConvexHull)
            print("\n\n")

            print("Depth values ; ", depthValues)

            currMinDepth = min(depthValues)
            currMaxDepth = max(depthValues)
            
            print("currMinDepth = ", currMinDepth)
            print("currMaxDepth = ", currMaxDepth)
            
            
            pointsToComputeConvexHull = v0ProjectionToComputeConvexHull + v1ProjectionToComputeConvexHull + v2ProjectionToComputeConvexHull 
            
            # print(pointsToComputeConvexHull)
            # print(len(pointsToComputeConvexHull))
            # # Extract x and y coordinates
            # x = [p[0]/10000 for p in pointsToComputeConvexHull]
            # y = [p[1]/10000 for p in pointsToComputeConvexHull]
            
            # for i in range(0, len(x)):
            #     print(x[i], y[i])
           

            # # Create the plot
            # plt.scatter(x, y)

            # # Add labels and title
            # plt.xlabel("X-axis")
            # plt.ylabel("Y-axis")
            # plt.title("2D Point Visualization")

            # # Show the plot
            # plt.show()
            
            # exit()    

            # pointsToComputeConvexHull = [projection0, projection1, projection2]
            hullCons, hullCornerPoints = computeConvexHull(pointsToComputeConvexHull)
            print(hullCornerPoints)

            # Remove all whitespaces from the list elements
            hullCornerPoints = [item.strip() for item in hullCornerPoints]

            # Convert each string element to a fraction
            hullCornerPoints = [eval(item) for item in hullCornerPoints]

            # Print the actual floating point values
            for item in hullCornerPoints:
                print(item)

            
            
            #now we have to move the corner points to the top left corner of the pixels
            #and then we compute the convex hull again
            #while moving y moves to top and x moves back

            newHullCornerPoints = [math.floor(x) if i % 2 == 0 else math.ceil(x) for i, x in enumerate(hullCornerPoints)]  
            print(newHullCornerPoints)

            newCornerPointsToComputeConvexHull = [newHullCornerPoints[i:i+2] for i in range(0, len(newHullCornerPoints), 2)]
            
            print(newCornerPointsToComputeConvexHull)

            polyhedraOfHull = computeConvexHull2(newCornerPointsToComputeConvexHull)
            

            # print("\n\n")
            # print(getHullCornerPoints(polyhedraOfHull))
            # print(polyhedraOfHull.constraints())
            
            polyhedraOfHull.intersection_assign(canvasPolyhedra)
            # print(polyhedraOfHull.constraints())
            # print(getHullCornerPoints(polyhedraOfHull))
            # print(polyhedraOfHull.is_empty())
            # exit()

            if polyhedraOfHull.is_empty():
                print("Empty Intersection Triangle")
                numberOfEmptyIntersectionTriangles += 1
                continue
            
            
            #compute pixels that are inside the intersection of all vertices regions
            v0RegionPd = computeConvexHull3(v0ProjectionToComputeConvexHull) 
            v1RegionPd = computeConvexHull3(v1ProjectionToComputeConvexHull)
            v2RegionPd = computeConvexHull3(v2ProjectionToComputeConvexHull)
            
            print("v0RegionPd: ", v0RegionPd.minimized_constraints())
            print("v1RegionPd: ", v1RegionPd.minimized_constraints())
            print("v2RegionPd: ", v2RegionPd.minimized_constraints())
            
            
            v0RegionPd.intersection_assign(v1RegionPd)
            print("v0RegionPd.intersection_assign(v1RegionPd): ", v0RegionPd.minimized_constraints())
            v0RegionPd.intersection_assign(v2RegionPd)
            print("v0RegionPd.intersection_assign(v2RegionPd): ", v0RegionPd.minimized_constraints())
            
            commonPixelExist = False
            
            if v0RegionPd.is_empty():
                print("No Common Pixels")
            else:
                commonPixelExist = True
                print("Intersecting region corner points")
                print(getHullCornerPoints(v0RegionPd))
                
                
                
                
                
                
            


            
            hullCornerPoints2 = getHullCornerPoints(polyhedraOfHull)
            
            
            
            print(hullCornerPoints2)
            # print(hullCons2)

           

           

            #replace all the strings with "closure_" with ""
            hullCornerPoints2 = [item.replace("closure_","") for item in hullCornerPoints2] 

             # Remove all whitespaces from the list elements
            hullCornerPoints2 = [item.strip() for item in hullCornerPoints2]

           

            # Convert each string element to a fraction
            hullCornerPoints2 = [eval(item) for item in hullCornerPoints2]

            # Print the actual floating point values
            for item in hullCornerPoints2:
                print(item)

            
            # Extract elements at positions 0, 2, 4, ...
            elements_at_even_positions = [hullCornerPoints2[i] for i in range(0, len(hullCornerPoints2), 2)]
            elements_at_odd_positions = [hullCornerPoints2[i] for i in range(1, len(hullCornerPoints2), 2)]

            # Find the minimum and maximum
            minimumX = min(elements_at_even_positions)
            maximumX = max(elements_at_even_positions)
            minimumY = min(elements_at_odd_positions)
            maximumY = max(elements_at_odd_positions)


            print("Minimum:", minimumX)
            print("Maximum:", maximumX)
            print("Minimum:", minimumY)
            print("Maximum:", maximumY)

            print(polyhedraOfHull.constraints())

            # p, h = computeConvexHull2([[0,0],[1,1],[1,0],[0,1]])
            # print(h)
            # print(p.constraints())
            # print(polyhedraOfHull.constraints())

            
            #for each pixel which is inside the min-max bounding box check whether the center of the pixel is
            #inside the convex hull or not
            #if it is inside then we have to compute the depth value for the pixel
            
            
            pixelToColour = []
            commonPixelsToColour = []
            depthOfPixels = dict()
            depthOfPixels.clear()

            for y in range(int(minimumY), int(maximumY+1)):
                for x in range(int(minimumX), int(maximumX+1)):
                    centerX = str(x) + "5"
                    centerY = str(y) + "5"
                    # print(x, y , centerX, centerY)

                    pdTemp = NNC_Polyhedron(2,'empty')
                    pdTemp.add_generator(point(int(centerX)*px+int(centerY)*py, pow(10,1)))

                    if(polyhedraOfHull.contains(pdTemp)):  

                        pixelToConsider = [x, 49-(y+1)]
                        # currDepths = gurobiPixelDepths1.computeDepthsOfPixels(pixelToConsider, currTriangle,v0Vertex, v1Vertex, v2Vertex,
                        #                                    currAbsGroupRegionCons, currRegionMinMaxValues, 
                        #                                     currRegionCornerPoints)


                        print("Pixel is inside the convex hull")
                        # exit(0)

                        print("(",x, 49-y,")", end=" ")
                        if commonPixelExist and v0RegionPd.contains(pdTemp):
                            # if v0RegionPd.contains(pdTemp):
                            commonPixelsToColour.append(pixelToConsider)  
                        else: 
                            pixelToColour.append(pixelToConsider) 
                    # else:
                    #     print("Pixel is outside the convex hull")
                        
            print(pixelToColour)
            
            
            dataToGenerateCurrTriangleIntervalImage = [currMinDepth, currMaxDepth, pixelToColour, commonPixelsToColour, depthOfPixels]
            
            # exit()
            
            # print(originalPixelToColor)

            # print(len(pixelToColour), len(originalPixelToColor))

            
            # #check whether the pixles in originalPixelTOCOlor also exist in pixelToColour
            # for item in originalPixelToColor:
            #     if item not in pixelToColour:
            #         print("Pixel not in pixelToColour", item)
            # print("\n\n") 
            # for item in pixelToColour :
            #     if item not in originalPixelToColor:
            #         print("Pixel not in pixelToColour", item)
            
            singleTriangleIntervalImageData[currTriangle] = dataToGenerateCurrTriangleIntervalImage

        elif( v0Vertex[2] > currRegionMinMaxValues[5]-1  and v1Vertex[2] > currRegionMinMaxValues[5]-1 and
            v2Vertex[2] > currRegionMinMaxValues[5]-1):
            print("Triangle is fully in backside of the region")
            numberOfFullyBacksideTriangles += 1

        else:
            print("Triangle is not fully in front of the region,  intersecting with the region")
            numberOfIntersectingTriangles += 1
            intersectingTriangles.append(currTriangle)
        


   
    
    print("\n Single Triangle Interval Image Data")
    print(singleTriangleIntervalImageData)
    
    print(globalIntervalImageP3)
    prepareGlobalIntervalImageP3()
    # print(globalIntervalImageP3)
    # prepareGlobalIntervalImageP3()
    print(globalIntervalImageP3)
    
    print("Number of intersecting Triangles: ", numberOfIntersectingTriangles)
    print("Number of fully backside Triangles: ", numberOfFullyBacksideTriangles)
    print("Number of empty intersection Triangles: ", numberOfEmptyIntersectionTriangles)
    print("Intersecting triangles = ", intersectingTriangles)

    
    if len(intersectingTriangles)> 0:
        intersectingTrianglesIntImage =  singleTriangleInvRegions31_P3_1.computePixelIntervalsOfIntersectingTriangles(
            intersectingTriangles,currAbsGroupName, currAbsGroupRegionCons)

        if  len(intersectingTrianglesIntImage) != 0:
            print("Store intersecting triangle data  to singleTriangleIntervalImageData")
            print(intersectingTrianglesIntImage)
            updateGlobalIntervalImageForIntersectingTriangles(intersectingTrianglesIntImage)

    print("Computed interval image for intersecting triangles")        

    

    generate_vnnlib_files2(globalIntervalImageP3)



    print("running alpha beta crown for each output")
    os.environ["MKL_SERVICE_FORCE_INTEL"]="1"
    os.environ["MKL_THREADING_LAYER"]="GNU"
    
    y0OutFile = open("abc_out_y0.txt","w")
    y0OutFile.write("UNSAT")
    y0OutFile.close()
    y1OutFile = open("abc_out_y1.txt","w")
    y1OutFile.write("UNSAT")
    y1OutFile.close()
    y2OutFile = open("abc_out_y2.txt","w")
    y2OutFile.write("UNSAT")
    y2OutFile.close()
    
    print(datetime.now())
    
    # tempString = "python /home/habeeb/projectDrone2/alpha-beta-CROWN-main/complete_verifier/abcrown.py  --onnx_path OGmodel_pb_converted.onnx --vnnlib_path prop_y0.vnnlb --device cpu --results_file 'abc_out_y0.txt' --no_incomplete"
    
    tempString = "python /home/habeeb/project2/alpha-beta-CROWN-mainfromLap/complete_verifier/abcrown.py  --onnx_path OGmodel_pb_converted.onnx --vnnlib_path prop_y0.vnnlb --device cpu --results_file 'abc_out_y0.txt' --no_incomplete"
    print(tempString)    
    os.system(tempString)
    y0OutFile = open("abc_out_y0.txt")
    y0Out = y0OutFile.read()
    print(y0Out)
    
    # tempString = "python /home/habeeb/projectDrone2/alpha-beta-CROWN-main/complete_verifier/abcrown.py  --onnx_path OGmodel_pb_converted.onnx --vnnlib_path prop_y1.vnnlb --device cpu --results_file 'abc_out_y1.txt' --no_incomplete"
   
    tempString = "python /home/habeeb/project2/alpha-beta-CROWN-mainfromLap/complete_verifier/abcrown.py  --onnx_path OGmodel_pb_converted.onnx --vnnlib_path prop_y1.vnnlb --device cpu --results_file 'abc_out_y1.txt' --no_incomplete"
    print(tempString)    
    os.system(tempString)
    y1OutFile = open("abc_out_y1.txt")
    y1Out = y1OutFile.read()
    print(y1Out)
    
    # tempString = "python /home/habeeb/projectDrone2/alpha-beta-CROWN-main/complete_verifier/abcrown.py  --onnx_path OGmodel_pb_converted.onnx --vnnlib_path prop_y2.vnnlb --device cpu --results_file 'abc_out_y2.txt' --no_incomplete"
   
    tempString = "python /home/habeeb/project2/alpha-beta-CROWN-mainfromLap/complete_verifier/abcrown.py  --onnx_path OGmodel_pb_converted.onnx --vnnlib_path prop_y2.vnnlb --device cpu --results_file 'abc_out_y2.txt' --no_incomplete"
    print(tempString)    
    os.system(tempString)
    y2OutFile = open("abc_out_y2.txt")
    y2Out = y2OutFile.read()
    print(y2Out)
    
    print(y0Out, y1Out,y2Out)
    print("sleeping")
    
    
    
    # print("Starting DeepPoly")
    # print(str(datetime.now()))
    deepPolyOutputs2 = []
    # deepPolyOutputs2 = interval_image_translator_3habeeb.runDeepPoly()
    deepPolyOutputs = []
    
    if(y0Out == "sat"):
        deepPolyOutputs.append(0)
        
        
    if(y1Out == "sat"):
        deepPolyOutputs.append(1)
        
    
        
    if(y2Out == "sat"):
        deepPolyOutputs.append(2)
    
    
    
    
    
    y0OutFile.close()
    y1OutFile.close()
    y2OutFile.close()


    fromSplitRegion = 0
    currGroupName = currAbsGroupName
    if int(fromSplitRegion) == 0:
        for o in range(0,len(deepPolyOutputs)):
            print(deepPolyOutputs[o])
            # if(deepPolyOutputs[o] == iisc_net_dnnoutput):
            #     print("skipping iisc_net_dnnoutput, will add later")
            #     continue
            environment.absStack.append(currGroupName+str(deepPolyOutputs[o]))
            
            currentNodeParentName = currGroupName[:currGroupName.rfind("_")]
            currentNode = currGroupName+str(deepPolyOutputs[o])
            currentNodeParent = anytree.find(environment.A, filter_=lambda node: node.name==currentNodeParentName)
            currentNode = anytree.Node(currentNode, parent=currentNodeParent)

            print("currentNodeName = ",currentNode)
            print("currentNodeParentName = ",currentNodeParentName)
        
        # environment.absStack.append(currGroupName+str(iisc_net_dnnoutput))
        
        # currentNodeParentName = currGroupName[:currGroupName.rfind("_")]
        # currentNode = currGroupName+str(iisc_net_dnnoutput)
        # currentNodeParent = anytree.find(environment.A, filter_=lambda node: node.name==currentNodeParentName)
        # currentNode = anytree.Node(currentNode, parent=currentNodeParent)

        
            
        # environment.absStack.append(currGroupName+str(og_net_dnnoutput))
    else:
        return deepPolyOutputs
    
                   





                    
# currAbsGroupName = "A_"
# currAbsGroupRegionCons = environment.initCubeCon
# currRegionMinMaxValues = environment.initRegionMinMaxValues
# currRegionCornerPoints = environment.initRegionCornerPoints

# computeIntervalImage_P3(currAbsGroupName, currAbsGroupRegionCons, currRegionMinMaxValues, currRegionCornerPoints)











