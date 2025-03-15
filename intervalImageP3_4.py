import numpy as np
import os
import anytree
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt

import math
from scipy.spatial import ConvexHull
import cv2
import onnx
import onnxruntime
import sys

from pyparma import *

import environment
import singleTriangleInvRegions31_P3_1
# import pythonRenderAnImage2
import renderAnImageForP3
import gurobiPixelDepths1

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

mProj = [
        [2 * n / (r - l), 0, 0, 0],
        [0,2 * n / (t - b),0,0],
        [(r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n), -1 ],
        [0,0,-2 * f * n / (f - n),0]
    ]

def edgeFunction(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])  


def computeOutcodeAtPos2( inx, iny,inz):
    
    
    outx   = inx * mProj[0][0] + iny * mProj[1][0] + inz * mProj[2][0] +  mProj[3][0]
    outy   = inx * mProj[0][1] + iny * mProj[1][1] + inz * mProj[2][1] +  mProj[3][1] 
    outz   = inx * mProj[0][2] + iny * mProj[1][2] + inz * mProj[2][2] +  mProj[3][2] 
    w      = inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3] 
    
    outValueToReturn = [outx, outy, outz]
  
        
    return outValueToReturn, w

def getDNNOutput_onnx(inputImage,networkName):
    # dnnOutput = 1
    # model = onnx.load(networkName)

    image = cv2.imread(inputImage)  
    # print(image.shape)  
    image = cv2.resize(image, (49, 49)).copy()
    np.set_printoptions(threshold=sys.maxsize)
    # print("\n\n")
    
    # print(image)
    
    # print("\n------------\n")

    if networkName == "iisc_net1.onnx":
        a, b, c = image.shape
        image = image.reshape(1, c,b,a)
        # print(image.shape)
    else:
        a, b, c = image.shape
        # print(a,b,c)
        # exit()
        image = image.reshape(1, a,b,c)
        # print(image.shape)
        # 

    image = image.astype(np.float32) / 255.0
    # image2 = tf.convert_to_tensor(image)
    
    # print("\n\n")
    
    # print(image)
    
    # print("\n\n")
    
    # exit()
    

    session = onnxruntime.InferenceSession(networkName)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # print(input_name)
    # print(output_name)

    result = session.run([output_name], {input_name: image})
    # print("result = ", result)
    dnnOutput  = np.argmax(np.array(result).squeeze(), axis=0)
    # print("networkName = ", networkName)
    # print("dnnOutput = ", dnnOutput)
    return dnnOutput 





def pixelValue(point, w):
    t0 = point[0]/w
    t1 = point[1]/w
    t2 = point[2]/w
    
    # print("pixel values with one minus")
    # print(((t0 + 1) * 0.5 * imageWidth),((1 - (t1 + 1) * 0.5) * imageHeight), t2)
    originalPixel = [((t0 + 1) * 0.5 * imageWidth),(( (t1 + 1) * 0.5) * imageHeight), t2]
    # print("pixel value without minus, currently used for the computation = ",originalPixel  )
    
    raster0 = min(imageWidth - 1, int((t0 + 1) * 0.5 * imageWidth))
    raster1 = min(imageHeight - 1, int((1 - (t1 + 1) * 0.5) * imageHeight))
    raster2 = t2

    # print("pixel values actual = ",[raster0, raster1, raster2]  )
    # print("\n\n")
    
    return  originalPixel, [raster0, raster1, raster2]

def pixelValue2(point, w):
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



def getConvexHullPolyhedron(points):
    px = Variable(0)
    py = Variable(1)
    
    pd =NNC_Polyhedron(2,'empty')
    
    # print("computation reached")
    # print(points)
    
    for p in points: 
        # print(p)       
        pd.add_generator(point(p[0]*px+p[1]*py, pow(10,10)))
    
    return pd

def getConvexHullPolyhedron2(points):
    px = Variable(0)
    py = Variable(1)
    
    pd =NNC_Polyhedron(2,'empty')
    
    # print("computation reached")
    # print(points)
    
    for p in points: 
        # print(p)       
        pd.add_generator(point(p[0]*px+p[1]*py, pow(10,10)))
    
    return pd

def getConvexHullPolyhedronWithIntPoints(points):
    px = Variable(0)
    py = Variable(1)
    
    pd =NNC_Polyhedron(2,'empty')
    
    for p in points:        
        pd.add_generator(point(p[0]*px+p[1]*py))
    
    return pd

def computeConvexHull(points):
    px = Variable(0)
    py = Variable(1)
    
    pd =NNC_Polyhedron(2,'empty')
    
    for p in points:        
        pd.add_generator(point(p[0]*px+p[1]*py, pow(10,4)))
    
    # print(pd.constraints())
    
    gs = pd.generators()# // Use ph.minimized_generators() to minimal set of points for the polytope

    # print(gs)

    vertString = str(gs)
    vertString = vertString.replace("Generator_System {","").replace("}","").replace("point","").replace("(","").replace(")","")
    cornerPoints = vertString.split(",")
    # print(cornerPoints)  

    return pd.constraints(), cornerPoints


def getHullCornerPoints(hullCons):
    gs = hullCons.minimized_generators()# // Use ph.minimized_generators() to minimal set of points for the polytope

    # print(gs)

    vertString = str(gs)
    vertString = vertString.replace("Generator_System {","").replace("}","").replace("point","").replace("(","").replace(")","")
    cornerPoints = vertString.split(",")
    # print(cornerPoints)  

    return cornerPoints


def prepareGlobalIntervalImageP3():
    
    #for each triangle present in the singleTriangle Interval image
    #for each pixel
    #if the pixel present in the global interval image
    #if the current pixel depth interval is less than the global depth then replace color intervals
    #if it overlaps then expand both the depht and colour intervals   
    
    # print("globalIntervalImageP3 = ", globalIntervalImageP3)

    for ct in singleTriangleIntervalImageData:
        currTriangleData = singleTriangleIntervalImageData[ct]
        
        # print(nvertices[ct*3], nvertices[ct*3 + 1], nvertices[ct*3+2])
        currTriangleColour = [ vertColours[nvertices[ct*3]*3+0], vertColours[nvertices[ct*3 + 1]*3+1],
                              vertColours[nvertices[ct*3+2]*3+2]]
    
        # print("currTriangleColour = ",currTriangleColour)
        # print("currTriangleData = ",currTriangleData)
        # sleep(0.5)
        currTriangleMinDepth = currTriangleData[0]
        currTriangleMaxDepth = currTriangleData[1]
        for currPixel in currTriangleData[2]:
            # print(currPixel, end=" ")
            
            pixelIndex = currPixel[1]*imageWidth+currPixel[0]
            
            
                
            newDataToStore = [currTriangleMinDepth, 1000,
                             min(1,currTriangleColour[0]*255), max(1,currTriangleColour[0]*255),
                             min(25,currTriangleColour[1]*255), max(25,currTriangleColour[1]*255),
                             min(24,currTriangleColour[2]*255), max(24,currTriangleColour[2]*255)]
            currTriangleMaxDepth = 1000

            # if pixelIndex == 520:
            #     print("############P~~520")
            #     print("currTriangleData = ",ct)
            #     print("Data = ", newDataToStore)
            
            if globalIntervalImageP3.get(pixelIndex):
                # print(" Data Already there")
                currGlobalData = globalIntervalImageP3[pixelIndex]
                currMinD  = currGlobalData[0]
                currMaxD  = currGlobalData[1]
               
                
                if currTriangleMaxDepth < currMinD:
                    #replace the colours
                    globalIntervalImageP3[pixelIndex] = newDataToStore
                elif currTriangleMinDepth > currMaxD:
                    pass
                # else:
                elif currTriangleMinDepth < currMaxD:
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
                    # print("Union the current data")
                    # print("union data = ", globalIntervalImageP3[pixelIndex])
            
            else:
                # 
               
                    
                #     print("Data = ", newDataToStore)
                globalIntervalImageP3[pixelIndex] = newDataToStore


        # print("globalIntervalImageP3 = ", globalIntervalImageP3)      
        for currPixel in currTriangleData[3]:
            # print("Common Pixel EXIST")
            # print(currPixel, end=" ")
            pixelIndex = currPixel[1]*imageWidth+currPixel[0]
            newDataToStore = [currTriangleMinDepth, currTriangleMaxDepth,
                             currTriangleColour[0]*255, currTriangleColour[0]*255,
                             currTriangleColour[1]*255, currTriangleColour[1]*255,
                             currTriangleColour[2]*255, currTriangleColour[2]*255]
            # print("new data to store = ", newDataToStore)


           

            if globalIntervalImageP3.get(pixelIndex):
                
                currGlobalData = globalIntervalImageP3[pixelIndex]
                currMinD  = currGlobalData[0]
                currMaxD  = currGlobalData[1]

                
                
                if currTriangleMaxDepth < currMinD:
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
                # print("New Data")
                globalIntervalImageP3[pixelIndex] = newDataToStore

    # print("\n\nglobalIntervalImageP3 = ", globalIntervalImageP3[520])
    # print("globalIntervalImageP3 = ", globalIntervalImageP3)  

 
def updateGlobalIntervalImageForIntersectingTriangles(intersectingTrianglesIntImage):

    # print(globalIntervalImageP3)

    for ct in intersectingTrianglesIntImage:
        currTriangleData  = intersectingTrianglesIntImage[ct]
        # print(currTriangleData)
        # print(len(currTriangleData))
        
        # print("currTriangleData = ",ct)
        # print("Intersecting triangle data update")

        # currTriangleColour = [ vertColours[nvertices[ct*3]], vertColours[nvertices[ct*3 + 1]],
        #                       vertColours[nvertices[ct*3+2]]]
       
        for currPixel in currTriangleData:

            
            
            currPixelData = currTriangleData[currPixel]

            # if currPixel == 1225:
            #     print("############2P~~1225")
            #     print("currTriangle = ",ct)
            #     print("Data = ", currPixelData)


            
                    
            if globalIntervalImageP3.get(currPixel):
                
                    
                currGlobalData = globalIntervalImageP3[currPixel]
                currMinD  = currGlobalData[0]
                currMaxD  = currGlobalData[1]

                currTriangleMaxDepth = currPixelData[7]
                currTriangleMinDepth = currPixelData[6]
                if currTriangleMaxDepth < currMinD:
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
                # if currPixel == 1225:
                #     print("new datat")
                newDataToStore = [currPixelData[6], currPixelData[7],
                             currPixelData[0], currPixelData[1],
                             currPixelData[2], currPixelData[3],
                             currPixelData[4], currPixelData[5]]
                
                globalIntervalImageP3[currPixel] = newDataToStore

              

            

            # currTriangleMinDepth = currTriangleData[currPixel][6]
            # currTriangleMaxDepth = currTriangleData[currPixel][7]

            # print(currPixel,": ", currTriangleMinDepth, currTriangleMaxDepth)

               


def generate_vnnlib_files2(globalIntervalImageP3):
    
    
    tempString = ""
    tempList = []

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
        
        
            tempList.append(globalIntervalImageP3[i][2])
            tempList.append(globalIntervalImageP3[i][3])
            tempList.append(globalIntervalImageP3[i][4])
            tempList.append(globalIntervalImageP3[i][5])
            tempList.append(globalIntervalImageP3[i][6])
            tempList.append(globalIntervalImageP3[i][7])

            
        
        
        else:
            tempString += "(assert (>= X_"+str(i*3+0)+" "+str(1/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+0)+" "+str(1/255)+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+1)+" "+str(25/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+1)+" "+str(25/255)+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+2)+" "+str(24/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+2)+" "+str(24/255)+"))\n"

            tempList.append(1)
            tempList.append(1)
            tempList.append(25)
            tempList.append(25)
            tempList.append(24)
            tempList.append(24)
            
        
        
    
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


    f0 = open("globalMin.txt", "w")
    f1 = open("globalMax.txt", "w")
    for i in range(0,49*49):
        
        # if i == 1056:
        #     print("1056")
        #     print(globalIntervalImageP3[i])
        #     print(tempList[i*6+0])
        #     print(tempList[i*6+1])
        #     print(tempList[i*6+2])
        #     print(tempList[i*6+3])
        #     print(tempList[i*6+4])
        #     print(tempList[i*6+5])

        f0.write(str(int(tempList[i*6+0]))+"\n")
        f1.write(str(int(tempList[i*6+1]))+"\n")
        f0.write(str(int(tempList[i*6+2]))+"\n")
        f1.write(str(int(tempList[i*6+3]))+"\n")
        f0.write(str(int(tempList[i*6+4]))+"\n")
        f1.write(str(int(tempList[i*6+5]))+"\n")
        
        # if i in finalGlobalIntervalImage:
          
        #     f0.write(str(int(finalGlobalIntervalImage[i][0]))+"\n")
        #     f0.write(str(int(finalGlobalIntervalImage[i][2]))+"\n")
        #     f0.write(str(int(finalGlobalIntervalImage[i][4]))+"\n")
        # else:
        #     f0.write(str("1\n25\n24\n"))
            
    f0.close() 
    f1.close() 

    del tempString2
    del tempString
    

def computeIntervalImage_P3(currAbsGroupName, currAbsGroupRegionCons, currRegionMinMaxValues, 
                            currRegionCornerPoints,fromSplitRegion=0):   

##
    numberOfIntersectingTriangles = 0
    numberOfFullyBacksideTriangles = 0
    numberOfEmptyIntersectionTriangles = 0

    singleTriangleIntervalImageData.clear()  
    globalIntervalImageP3.clear()
    # print("globalIntervalImageP3 = ", globalIntervalImageP3)

    px = Variable(0)
    py = Variable(1)

    canvasPolyhedra = NNC_Polyhedron(2,'empty')
    canvasPolyhedra.add_generator(point(0*px+0*py))
    canvasPolyhedra.add_generator(point(49*px+0*py))
    canvasPolyhedra.add_generator(point(0*px+49*py))
    canvasPolyhedra.add_generator(point(49*px+49*py))


    # print(canvasPolyhedra.constraints())

    # print("currRegionMinMaxValues =  ", currRegionMinMaxValues)
    # print("currRegionCornerPoints =  ", currRegionCornerPoints)

    intersectingTriangles = []





    currRegionConsString1 = str(currAbsGroupRegionCons)
    currRegionConsString1 = currRegionConsString1.replace("And(","")
    currRegionConsString1 = currRegionConsString1.replace(")","")
    currRegionConsString1 = currRegionConsString1.replace("\n","")
    currRegionConsString1 = currRegionConsString1.replace("  ","")
    
    
    regionConsListToGurobi = currRegionConsString1.split(",")
    # print("regionConsListToGurobi = ", regionConsListToGurobi)

    for currTriangle in range(0,environment.numOfTriangles):
    # for currTriangle in range(0,100):
        # if currTriangle != 155 and currTriangle != 246:
        #     continue
        
        # if currTriangle != 31493:
        #     continue
        # if currTriangle != 2 and currTriangle !=6:
        #     continue

        # tLIst = [0, 1, 6, 7, 8, 9, 18, 19, 22, 23, 376, 377, 433, 438, 439, 448, 454, 457, 459, 460, 491, 492, 496,
        #           497, 513, 514, 520, 521, 570, 573, 575, 576, 609, 610, 615, 616, 647, 650, 652, 653, 699, 700,
        #             705, 706, 727, 728, 732, 735, 745, 746, 761, 764, 767, 769, 834, 835, 840, 841, 873, 876, 878, 
        #             879, 923, 925, 929, 930, 965, 967, 969, 971, 973, 984, 992, 993, 1108, 1109, 1112, 1118, 1119,
        #               1120, 1123, 1240, 1241, 1249, 1266, 1269, 1297, 1298, 1300, 1301, 1302, 1305, 1308, 1309, 1312, 
        #               1315, 1318, 1321, 1322, 1337, 1338, 1342, 1343, 1357, 1358, 1366, 1369, 1373, 1374, 1382, 1385,
        #                 1389, 1390, 1393, 1394, 1398, 1399, 1409, 1410, 1414, 1415, 1425, 1426, 1824, 1825, 1833, 1834, 
        #                 1865, 1866, 1898, 1899, 1941, 1942, 1943, 1944, 1954, 1956, 1961, 1962, 2053, 2054, 2059, 2061]
        # if currTriangle in tLIst:
        #     continue

        # if currTriangle not in [2]:
        #     continue

       
        # print("\n-----------------\ncurrernt Triangle::: ", currTriangle)
       
        # if currTriangle % 5000 == 0:
        #     print(currTriangle)
        #     print("\n-----------------\ncurrernt Triangle::: ", currTriangle)
        vertex0 = nvertices[currTriangle*3+0]
        vertex1 = nvertices[currTriangle*3+1]
        vertex2 = nvertices[currTriangle*3+2]
        currTriangleVertices = [vertex0, vertex1,vertex2]

        v0Vertex = [vertices[currTriangleVertices[0]*3+0], vertices[currTriangleVertices[0]*3+1],vertices[currTriangleVertices[0]*3+2] ]
        v1Vertex = [vertices[currTriangleVertices[1]*3+0], vertices[currTriangleVertices[1]*3+1],vertices[currTriangleVertices[1]*3+2] ]
        v2Vertex = [vertices[currTriangleVertices[2]*3+0], vertices[currTriangleVertices[2]*3+1],vertices[currTriangleVertices[2]*3+2] ]

        # print("Current Triangle Info")
        # print("Triangle Id: ", currTriangle)
        # print("Vertices index : ", currTriangleVertices)
        # print("Vertices : ", v0Vertex, "\n", v1Vertex, "\n", v2Vertex)
        
        
        # if Counter(v0Vertex) == Counter(v1Vertex) or Counter(v0Vertex) == Counter(v2Vertex) or Counter(v1Vertex) == Counter(v2Vertex):
        #     # print(v0Vertex, v1Vertex, v2Vertex)
        #     print("Error triangle")
        #     print("Counter(v0Vertex) ", Counter(v0Vertex))
        #     print("Counter(v1Vertex) ", Counter(v1Vertex))
        #     print("Counter(v2Vertex) ", Counter(v2Vertex))
        #     # print(currTriangle)
        #     continue

        if v0Vertex == v1Vertex or v0Vertex == v2Vertex or v1Vertex == v2Vertex:
            # print("Error triangle")
            # print("v0Vertex ", v0Vertex)
            # print("v1Vertex ", v1Vertex)
            # print("v2Vertex ", v2Vertex)
            continue
        
        # print("reached here")  
        # if(vertices[currTriangleVertices[0]*3+2] > posZp100+1 and vertices[currTriangleVertices[1]*3+2] > posZp100+1 and vertices[currTriangleVertices[2]*3+2] > posZp100+1 ):
        #     print("Invisible from region")
        #     return 0
        
        
        vertexProjectedPoints = dict()
        vertexProjectedPoints.clear()

        if( v0Vertex[2] <= currRegionMinMaxValues[4]-1  and v1Vertex[2] <= currRegionMinMaxValues[4]-1 and
            v2Vertex[2] <= currRegionMinMaxValues[4]-1):
            # print("Triangle is fully in front of the region")

            pointsToComputeConvexHull = []
            
            v0ProjectionToComputeConvexHull =[]
            v1ProjectionToComputeConvexHull =[]
            v2ProjectionToComputeConvexHull =[]

            depthValues = []

            internalProjections = []

            cornerPointProjections = []
            
            # print("currRegionCornerPoints: ", currRegionCornerPoints)
            
            for currCornerPoint in currRegionCornerPoints:
                # print("\n\n\nCurrent Corner Point: ", currCornerPoint)
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

                # print(pixel0, pixel1, pixel2)
                # print("\nProjected points: ", projection0, projection1, projection2)

                # originalPixelToColor = drawTriangle4(pixel0, pixel1, pixel2)

                if int(projection0[0]) >=0 and int(projection0[0]) < 49 and int(projection0[1]) >=0 and int(projection0[1]) < 49:
                    internalProjections.append([int(projection0[0]), math.ceil(projection0[1])])
                if int(projection1[0]) >=0 and int(projection1[0]) < 49 and int(projection1[1]) >=0 and int(projection1[1]) < 49:
                    internalProjections.append([int(projection1[0]), math.ceil(projection1[1])])
                if int(projection2[0]) >=0 and int(projection2[0]) < 49 and int(projection2[1]) >=0 and int(projection2[1]) < 49:
                    internalProjections.append([int(projection2[0]), math.ceil(projection2[1])])


                

                projection0[0] = int(projection0[0]*pow(10,10))
                projection0[1] = int(projection0[1]*pow(10,10))
                projection1[0] = int(projection1[0]*pow(10,10))
                projection1[1] = int(projection1[1]*pow(10,10))
                projection2[0] = int(projection2[0]*pow(10,10))
                projection2[1] = int(projection2[1]*pow(10,10))
                
                v0ProjectionToComputeConvexHull.append(projection0)
                v1ProjectionToComputeConvexHull.append(projection1)
                v2ProjectionToComputeConvexHull.append(projection2)

                depthValues.append(posZp - v0Vertex[2] )
                depthValues.append(posZp - v1Vertex[2])
                depthValues.append(posZp - v2Vertex[2])

                cornerPointProjections.append([projection0, projection1, projection2])

               

                # depthValues.append(1/(posZp - v0Vertex[2]))
                # depthValues.append(1/(posZp - v1Vertex[2]))
                # depthValues.append(1/(posZp - v2Vertex[2]))
            
                # print("current depth values: ", depthValues)
            
            # print("\n\n\n\n")
            # print("v0ProjectionToComputeConvexHull: ", v0ProjectionToComputeConvexHull)
            # print("v1ProjectionToComputeConvexHull:",v1ProjectionToComputeConvexHull)
            # print("v2ProjectionToComputeConvexHull: ",v2ProjectionToComputeConvexHull)
            # print("\n\n")

            # print("Depth values ; ", depthValues)

            currMinDepth = min(depthValues)
            currMaxDepth = max(depthValues)
            
            # print("\n\ncurrMinDepth = ", currMinDepth)
            # print("\n\ncurrMaxDepth = ", currMaxDepth)
            
            
            pointsToComputeConvexHull = v0ProjectionToComputeConvexHull + v1ProjectionToComputeConvexHull + v2ProjectionToComputeConvexHull 
            
            # print("pointsToComputeConvexHull: ",pointsToComputeConvexHull)
           
            # # print(len(pointsToComputeConvexHull))
            # # # Extract x and y coordinates
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


            fullConvexhull = getConvexHullPolyhedron(pointsToComputeConvexHull)
            # print(fullConvexhull.constraints())
            # print([eval(item) for item in getHullCornerPoints(fullConvexhull)] )

            # print("\n\n")
            # # print("Going to intersect with the canvas")
            
            
            # new_list = [sublist[:2] for sublist in pointsToComputeConvexHull]
            
            # scaled_list = [[x / 10**10 for x in sublist] for sublist in new_list]
            # points = np.array(new_list)
            # # Compute the convex hull
            # hull = ConvexHull(points)
            # print("points = ", points)
            # # Print the corner points of the convex hull
            # corner_points = points[hull.vertices]
            # print("Corner points of the convex hull:")
            # print(corner_points)
          
            
            
            

            fullConvexhull.intersection_assign(canvasPolyhedra)

            # print(fullConvexhull.minimized_constraints())
            
            

            if fullConvexhull.is_empty():
                # print("Empty Intersection Triangle")
                numberOfEmptyIntersectionTriangles += 1
                continue
            
            hullCornerPoints = getHullCornerPoints(fullConvexhull)
            # print(hullCornerPoints)

           
            #replace all the strings with "closure_" with ""
            hullCornerPoints = [item.replace("closure_","") for item in hullCornerPoints] 

             # Remove all whitespaces from the list elements
            hullCornerPoints = [item.strip() for item in hullCornerPoints]

           

            # Convert each string element to a fraction
            hullCornerPoints = [eval(item) for item in hullCornerPoints]

            # print(hullCornerPoints)
            # # Print the actual floating point values
            # for item in hullCornerPoints:
            #     print(item)

            # print("\n\n Move to the top left")

            #now we have to move the corner points to the top left corner of the pixels
            #and then we compute the convex hull again
            #while moving y moves to top and x moves back

            newHullCornerPoints = [math.floor(x) if i % 2 == 0 else math.ceil(x) for i, x in enumerate(hullCornerPoints)]  
            

            # newCornerPointsToComputeConvexHull = [newHullCornerPoints[i:i+2] for i in range(0, len(newHullCornerPoints), 2)]
            
            # print(newCornerPointsToComputeConvexHull)

            # print("new hull corner points for which we draw the diagram")
            # print(newHullCornerPoints)

            # # newHullCornerPoints = [0 if x < 0 else 48 if x > 48 else x for x in newHullCornerPoints]

            # print("modified new hull corner points for which we draw the diagram")
            # print(newHullCornerPoints)

            pointsToComputThefinalHull = [list(pair) for pair in zip(newHullCornerPoints[::2], newHullCornerPoints[1::2])]
            # print("pointsToComputThefinalHull: ", pointsToComputThefinalHull)

            for sublist in pointsToComputThefinalHull:
                if sublist[0] < 0:
                    sublist[0] = 0
                elif sublist[0] > 48:
                    sublist[0] = 48
                
                if sublist[1] < 1:
                    sublist[1] = 1
                elif sublist[1] > 49:
                    sublist[1] = 49

            # print("pointsToComputThefinalHull2: ", pointsToComputThefinalHull)

            #ensuring internal projections
            pointsToComputThefinalHull = pointsToComputThefinalHull + internalProjections

            # print("\npointsToComputeConvexHull3: ", pointsToComputThefinalHull)
            # print("\n")

            newCornerpointPolyhedra = getConvexHullPolyhedronWithIntPoints(pointsToComputThefinalHull)
            
            # print("newCornerpointPolyhedra: ", newCornerpointPolyhedra.minimized_constraints())




            ###Ensure positive area
            unique_points = np.unique(pointsToComputThefinalHull, axis=0)  # Remove duplicate points
    
            if len(unique_points) < 3:
                # print("Collinear points1!!!!!")
                continue
                

            # try:
            #     hull = ConvexHull(unique_points)
            #     print("Hull volume = ", hull.volume)  # In 2D, 'volume' stores the area
            # except Exception:  # Handles collinear points issue
            #     print("Collinear points2!!!!!")
            #     continue
            
            
           



            # Extract elements at positions 0, 2, 4, ...
            # elements_at_even_positions = [pointsToComputThefinalHull[i] for i in range(0, len(pointsToComputThefinalHull), 2)]
            # elements_at_odd_positions = [pointsToComputThefinalHull[i] for i in range(1, len(pointsToComputThefinalHull), 2)]

            elements_at_even_positions = [x[0] for x in pointsToComputThefinalHull]
            elements_at_odd_positions = [x[1] for x in pointsToComputThefinalHull]

            # Find the minimum and maximum
            minimumX = min(elements_at_even_positions)
            maximumX = max(elements_at_even_positions)
            minimumY = min(elements_at_odd_positions)
            maximumY = max(elements_at_odd_positions)


            # print("Minimum:", minimumX)
            # print("Maximum:", maximumX)
            # print("Minimum:", minimumY)
            # print("Maximum:", maximumY)


            # # minimumY =9
           

            # print("\n\n\nCommon pixel computation started\n")
            # print(cornerPointProjections)
            commonPixelExist = False
            commonPixelFlag = 1
            justACounter = 0
            cornerPointImageVertices = []
            for currCornerPointToConsider in cornerPointProjections:
                # print("Just a counter: ", justACounter)
                # print("currCornerPointToConsider: ", currCornerPointToConsider)
                justACounter += 1
                c0RegionPdTemp = getConvexHullPolyhedron2(currCornerPointToConsider)
                # print(v0RegionPdTemp.minimized_constraints()) 
                c0RegionPdTemp.intersection_assign(canvasPolyhedra)
                if not c0RegionPdTemp.is_empty():
                    c0FinalCornerPoints = getHullCornerPoints(c0RegionPdTemp)
                    #replace all the strings with "closure_" with ""
                    c0FinalCornerPoints = [item.replace("closure_","") for item in c0FinalCornerPoints] 
                    # Remove all whitespaces from the list elements
                    c0FinalCornerPoints = [item.strip() for item in c0FinalCornerPoints]
                    # Convert each string element to a fraction
                    c0FinalCornerPoints = [eval(item) for item in c0FinalCornerPoints]
                    c0FinalCornerPoints = [math.floor(x) if i % 2 == 0 else math.ceil(x) for i, x in enumerate(c0FinalCornerPoints)]  
            
                    c0FinalCornerPoints2 = [list(pair) for pair in zip(c0FinalCornerPoints[::2], c0FinalCornerPoints[1::2])]

                    # print("\nc0FinalCornerPoints2 = ", c0FinalCornerPoints2)
                    for sublist in c0FinalCornerPoints2:
                        if sublist[0] < 0:
                            sublist[0] = 0
                        elif sublist[0] > 48:
                            sublist[0] = 48
                        
                        if sublist[1] < 1:
                            sublist[1] = 1
                        elif sublist[1] > 49:
                            sublist[1] = 49

                    tempList = []
                    for i in c0FinalCornerPoints2:
                        tempList.append(i)
                    
                    if len(tempList) !=3:
                        # print("zero area image")
                        commonPixelFlag = 0
                        break
                    area =  edgeFunction(tempList[0], tempList[1], tempList[2])
                    # print("area : ", area)
                    if area == 0:
                        commonPixelFlag = 0
                        break

                    cornerPointImageVertices.append(tempList)
                    # print("c0FinalCornerPoints2 = ", c0FinalCornerPoints2)

                    # v0RegionPdTemp = getConvexHullPolyhedronWithIntPoints(c0FinalCornerPoints2)
                    # print(v0RegionPdTemp.minimized_constraints()) 
                    # print("v0FinalCornerPoints2 = ", v0FinalCornerPoints2)

            # print("cornerPointImageVertices: ", cornerPointImageVertices)

            firstCornerPD = NNC_Polyhedron(2,'empty')
            if len(cornerPointImageVertices) != 0 and commonPixelFlag == 1:
                firstCornerPD = getConvexHullPolyhedronWithIntPoints(cornerPointImageVertices[0])
                # print("firstCornerPD: ", firstCornerPD.minimized_constraints())
                for currImageToConsider in cornerPointImageVertices[1:]:
                    # print("currImageToConsider: ", currImageToConsider)
                    firstCornerPD.intersection_assign(getConvexHullPolyhedronWithIntPoints(currImageToConsider))
                    # print("firstCornerPD: ", firstCornerPD.minimized_constraints())
                    # print("\n")
                
                if firstCornerPD.is_empty():
                    # print("No Common Pixels")

                    pass
                else:
                    commonPixelExist = True
                    # print("Common Pixels Exist")
                    # print("Intersecting region corner points")
                    # print(getHullCornerPoints(firstCornerPD))



            # print("Common pixle computation finished\n\n\n")
            
            
            # firstCornerPD = getConvexHullPolyhedron(cornerPointProjections[0])
            # print("firstCornerPD: ", firstCornerPD.minimized_constraints())
            # for currImageToConsider in cornerPointProjections[1:]:
            #     print("currImageToConsider: ", currImageToConsider)
            #     firstCornerPD.intersection_assign(getConvexHullPolyhedron(currImageToConsider))
            #     print("firstCornerPD: ", firstCornerPD.minimized_constraints())
            #     print("\n")
            
            
            
            # if firstCornerPD.is_empty():
            #     # print("No Common Pixels")
            #     pass
            # else:
            #     commonPixelExist = True
            #     print("Common Pixels Exist")
            #     # print("Intersecting region corner points")
            #     # print(getHullCornerPoints(v0RegionPd))

            # print("\n\n") 

             
            
            
            
            
            
            
            
            
            
            
            # v0RegionPd = getConvexHullPolyhedron(v0ProjectionToComputeConvexHull)
            # v1RegionPd = getConvexHullPolyhedron(v1ProjectionToComputeConvexHull)
            # v2RegionPd = getConvexHullPolyhedron(v2ProjectionToComputeConvexHull)
            
            # # print("v0RegionPd: ", v0RegionPd.minimized_constraints())
            # # print("v1RegionPd: ", v1RegionPd.minimized_constraints())
            # # print("v2RegionPd: ", v2RegionPd.minimized_constraints())
            
            
            # v0RegionPd.intersection_assign(v1RegionPd)
            # # print("v0RegionPd.intersection_assign(v1RegionPd): ", v0RegionPd.minimized_constraints())
            # v0RegionPd.intersection_assign(v2RegionPd)
            # # print("v0RegionPd.intersection_assign(v2RegionPd): ", v0RegionPd.minimized_constraints())
            
           



            #for each pixel which is inside the min-max bounding box check whether the center of the pixel is
            #inside the convex hull or not
            #if it is inside then we have to compute the depth value for the pixel
            pixelToColour = []
            commonPixelsToColour = []
            pixelToColourForPrint =[]

            # print("currMinDepth: ", currMinDepth)
            # print("currMaxDepth: ", currMaxDepth)

            for y in range(int(minimumY), int(maximumY+1)):
                for x in range(int(minimumX), int(maximumX+1)):
                    centerX = str(x) + "5"
                    centerY = str(y) + "5"
                    # print(x, y , centerX, centerY)

                    pdTemp = NNC_Polyhedron(2,'empty')
                    pdTemp.add_generator(point(int(centerX)*px+int(centerY)*py, pow(10,1)))

                    currPixelToComputeDepth = [x, 49-(y+1)]

                    
                  
                    if(newCornerpointPolyhedra.contains(pdTemp)):   
                        

                        pixelToColourForPrint.append([x, 49-(y+1)])
                        # if x+(49*(49-(y+1))) == 520:
                        #     print("~~~~##~~~~520 : ", currTriangle)
                        #     currTriangleColour = [ vertColours[nvertices[currTriangle*3]*3+0], vertColours[nvertices[currTriangle*3 + 1]*3+1],
                        #       vertColours[nvertices[currTriangle*3+2]*3+2]]
                        #     print("currTriangleColour = ",currTriangleColour[0]*255)
                        #     print("(",x, 49-(y+1),")")
                        #     print("currMinDepth: ", currMinDepth)
                        #     print("currMaxDepth: ", currMaxDepth)

                        # print("currPixelToComputeDepth: ", currPixelToComputeDepth)
                       
                            
                        # print("(",x, 49-y,")", end=" ")
                        ######################################
                        #################TODO###############
                        if commonPixelExist and firstCornerPD.contains(pdTemp):
                            # if v0RegionPd.contains(pdTemp):
                            # if x+(49*(49-(y+1))) == 520:
                            #     print("Common Pixel 520")
                            #     print("(",x, 49-(y+1),")")
                            #     print("current triangle: ", currTriangle)
                            

                            # print("Pixel is inside the convex hull")
                           

                            currDepthsToProecess =  gurobiPixelDepths1.computeDepthsOfPixels(currPixelToComputeDepth, currTriangle,
                                                                v0Vertex, v1Vertex, v2Vertex, currAbsGroupRegionCons, 
                                                                 currRegionMinMaxValues, 
                                                                currRegionCornerPoints, regionConsListToGurobi, minMaxFlag =1)
                            
                            if currDepthsToProecess[0] == -1:
                                currDepthsToProecess[0] = currMinDepth
                            if currDepthsToProecess[1] == 1000:
                                currDepthsToProecess[1] = currMaxDepth

                            commonPixelsToColour.append([x, 49-(y+1), currDepthsToProecess])  
                            # print("currDepthsToProecess: ", currDepthsToProecess)
                            
                        else: 
                            currDepthsToProecess =  gurobiPixelDepths1.computeDepthsOfPixels(currPixelToComputeDepth, currTriangle,
                                                                v0Vertex, v1Vertex, v2Vertex, currAbsGroupRegionCons, 
                                                                 currRegionMinMaxValues, 
                                                                currRegionCornerPoints, regionConsListToGurobi, minMaxFlag =0)
                            if currDepthsToProecess[0] == -1:
                                currDepthsToProecess[0] = currMinDepth

                            pixelToColour.append([x, 49-(y+1), currDepthsToProecess]) 
                            # print("currDepthsToProecess: ", currDepthsToProecess)
                        ###################################################
                        # pixelToColour.append([x, 49-(y+1)])
                            
                        
                    # else:
                    #     print("Pixel is outside the convex hull ",  currPixelToComputeDepth)
                        
            # print("\n\npixelToColour : ",pixelToColour)
            # print("All pixels to colour = ",pixelToColourForPrint )
            # print("Common pixel to colour = ", commonPixelsToColour)
            # print("Pixel with background colour = ", pixelToColour)
            
            dataToGenerateCurrTriangleIntervalImage = [currMinDepth, currMaxDepth, pixelToColour, commonPixelsToColour]
            singleTriangleIntervalImageData[currTriangle] = dataToGenerateCurrTriangleIntervalImage

            
            

        elif( v0Vertex[2] > currRegionMinMaxValues[5]-1  and v1Vertex[2] > currRegionMinMaxValues[5]-1 and
            v2Vertex[2] > currRegionMinMaxValues[5]-1):
            # print("Triangle is fully in backside of the region")
            numberOfFullyBacksideTriangles += 1

        else:
            # print("Triangle is not fully in front of the region,  intersecting with the region")
            numberOfIntersectingTriangles += 1
            intersectingTriangles.append(currTriangle)
        
    
    # print("\n Single Triangle Interval Image Data")
    # print(singleTriangleIntervalImageData)
    

    # print("exit after the computation of front triangels")
    

    # print(globalIntervalImageP3)
    prepareGlobalIntervalImageP3()
    # print(globalIntervalImageP3)
    # prepareGlobalIntervalImageP3()
    # print(globalIntervalImageP3)
    
    # print("Number of intersecting Triangles: ", numberOfIntersectingTriangles)
    # print("Number of fully backside Triangles: ", numberOfFullyBacksideTriangles)
    # print("Number of empty intersection Triangles: ", numberOfEmptyIntersectionTriangles)
    # print("Intersecting triangles = ", intersectingTriangles)


   
    if(numberOfEmptyIntersectionTriangles == environment.numOfTriangles):
        return 0
    
   
    
    # print("globalIntervalImageP3 [520] = ", globalIntervalImageP3[520])

    if len(intersectingTriangles)> 0:
        intersectingTrianglesIntImage =  singleTriangleInvRegions31_P3_1.computePixelIntervalsOfIntersectingTriangles(
            intersectingTriangles,currAbsGroupName, currAbsGroupRegionCons)

        if  len(intersectingTrianglesIntImage) != 0:
            # print("Store intersecting triangle data  to singleTriangleIntervalImageData")
            # print(intersectingTrianglesIntImage)
            updateGlobalIntervalImageForIntersectingTriangles(intersectingTrianglesIntImage)

    # print("Computed interval image for intersecting triangles")        

    # print("globalIntervalImageP3 [520] = ", globalIntervalImageP3[520])

    generate_vnnlib_files2(globalIntervalImageP3)


    # print("Intersecting triangles = ", intersectingTriangles)
    # print("exiting")
    
 
    # print("Exiting")
    # exit()


    # print("running alpha beta crown for each output")
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
    
    # print(datetime.now())
    
    # tempString = "python /home/habeeb/projectDrone2/alpha-beta-CROWN-main/complete_verifier/abcrown.py  --onnx_path OGmodel_pb_converted.onnx --vnnlib_path prop_y0.vnnlb --device cpu --results_file 'abc_out_y0.txt' --no_incomplete"
    
    # tempString = "python /home/habeeb/project2/alpha-beta-CROWN-mainfromLap/complete_verifier/abcrown.py  --onnx_path OGmodel_pb_converted.onnx --vnnlib_path prop_y0.vnnlb --device cpu --results_file 'abc_out_y0.txt' --no_incomplete"
    # print(tempString)    
    tempString = f"python {environment.netInterpreterPath} --onnx_path {environment.networkName} --vnnlib_path prop_y0.vnnlb --device cpu --results_file 'abc_out_y0.txt' --no_incomplete > output_and_errors.log 2>&1"
    
    os.system(tempString)
    y0OutFile = open("abc_out_y0.txt")
    y0Out = y0OutFile.read()
    # print(y0Out)
    
    # tempString = "python /home/habeeb/projectDrone2/alpha-beta-CROWN-main/complete_verifier/abcrown.py  --onnx_path OGmodel_pb_converted.onnx --vnnlib_path prop_y1.vnnlb --device cpu --results_file 'abc_out_y1.txt' --no_incomplete"
    tempString = f"python {environment.netInterpreterPath} --onnx_path {environment.networkName} --vnnlib_path prop_y1.vnnlb --device cpu --results_file 'abc_out_y1.txt' --no_incomplete > output_and_errors.log 2>&1"    

    # tempString = "python /home/habeeb/project2/alpha-beta-CROWN-mainfromLap/complete_verifier/abcrown.py  --onnx_path OGmodel_pb_converted.onnx --vnnlib_path prop_y1.vnnlb --device cpu --results_file 'abc_out_y1.txt' --no_incomplete"
    # print(tempString)    
    
    os.system(tempString)
    y1OutFile = open("abc_out_y1.txt")
    y1Out = y1OutFile.read()
    # print(y1Out)
    
    # tempString = "python /home/habeeb/projectDrone2/alpha-beta-CROWN-main/complete_verifier/abcrown.py  --onnx_path OGmodel_pb_converted.onnx --vnnlib_path prop_y2.vnnlb --device cpu --results_file 'abc_out_y2.txt' --no_incomplete"
   
    # tempString = "python /home/habeeb/project2/alpha-beta-CROWN-mainfromLap/complete_verifier/abcrown.py  --onnx_path OGmodel_pb_converted.onnx --vnnlib_path prop_y2.vnnlb --device cpu --results_file 'abc_out_y2.txt' --no_incomplete"
    # print(tempString)    
    tempString = f"python {environment.netInterpreterPath} --onnx_path {environment.networkName} --vnnlib_path prop_y2.vnnlb --device cpu --results_file 'abc_out_y2.txt' --no_incomplete > output_and_errors.log 2>&1"
   
    os.system(tempString)
    y2OutFile = open("abc_out_y2.txt")
    y2Out = y2OutFile.read()
    # print(y2Out)
    
    # print(y0Out, y1Out,y2Out)
    # print("sleeping")
    
    
    
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
    
    
    
    
    # print("deepPolyOutputs = ",deepPolyOutputs)
    # print("from split region = ",fromSplitRegion)

    y0OutFile.close()
    y1OutFile.close()
    y2OutFile.close()

    posXp2 = currRegionCornerPoints[0][0]
    posYp2 = currRegionCornerPoints[0][1]
    posZp2 = currRegionCornerPoints[0][2]


    renderAnImageForP3.renderAnImage(posXp2,posYp2,posZp2,"testImage1")
    
    iisc_net_dnnoutput = getDNNOutput_onnx("images/testImage1.ppm",environment.networkName)
    
    # print("Test image dnn output = ", iisc_net_dnnoutput)
    # print("Deeppoly outputs",deepPolyOutputs)






    # fromSplitRegion = 0
    currGroupName = currAbsGroupName
    if int(fromSplitRegion) == 0:
        for o in range(0,len(deepPolyOutputs)):
            # print(deepPolyOutputs[o])
            if(deepPolyOutputs[o] == iisc_net_dnnoutput):
                # print("skipping iisc_net_dnnoutput, will add later")
                continue
            environment.absStack.append(currGroupName+str(deepPolyOutputs[o]))
            
            currentNodeParentName = currGroupName[:currGroupName.rfind("_")]
            currentNode = currGroupName+str(deepPolyOutputs[o])
            currentNodeParent = anytree.find(environment.A, filter_=lambda node: node.name==currentNodeParentName)
            currentNode = anytree.Node(currentNode, parent=currentNodeParent)

            # print("currentNodeName = ",currentNode)
            # print("currentNodeParentName = ",currentNodeParentName)
        
        environment.absStack.append(currGroupName+str(iisc_net_dnnoutput))
        currentNodeParentName = currGroupName[:currGroupName.rfind("_")]
        currentNode = currGroupName+str(iisc_net_dnnoutput)
        currentNodeParent = anytree.find(environment.A, filter_=lambda node: node.name==currentNodeParentName)
        currentNode = anytree.Node(currentNode, parent=currentNodeParent)

        # for i in [2,1,0]:
        #     environment.absStack.append(currGroupName+str(i))            
        #     currentNodeParentName = currGroupName[:currGroupName.rfind("_")]
        #     currentNode = currGroupName+str(i)
        #     currentNodeParent = anytree.find(environment.A, filter_=lambda node: node.name==currentNodeParentName)
        #     currentNode = anytree.Node(currentNode, parent=currentNodeParent)

        
            
        # environment.absStack.append(currGroupName+str(og_net_dnnoutput))
    else:
        return deepPolyOutputs
    






















