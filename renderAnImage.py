import environment

import numpy as np

import math
import array
import sys

from collections import Counter

from time import sleep

import camera
import scene


frameBuffer = dict()
depthBuffer = dict()

vertices = environment.vertices
nvertices = environment.nvertices
tedges = environment.tedges
imageWidth = environment.imageWidth
imageHeight = environment.imageHeight

errorTriangleList = []


# imageWidth = 1080
# imageHeight = 1080


# imageWidth = 49
# imageHeight = 49

canvasWidth = environment.canvasWidth
canvasHeight = environment.canvasHeight
focalLength = environment.focalLength
t= environment.t
b = environment.b
l = environment.l
r = environment.r
n = environment.n
f = environment.f
PI = 3.14159265358979323846
inchToMm = 25.4
filmApertureWidth = 0.9872
filmApertureHeight = 0.735

nearClippingPlane = camera.nearClippingPlane
farClippingPlane = camera.farClippingPlane
# t = 0
# b = 0
# l = 0
# r = 0
# n = 1
# f = 1000


# # OpenGL perspective projection matrix
mProj = [
        [2 * n / (r - l), 0, 0, 0],
        [0,2 * n / (t - b),0,0],
        [(r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n), -1 ],
        [0,0,-2 * f * n / (f - n),0]
    ]


# mProj = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

edges = []

    
def computeOutcodeAtPos(i,outcodeP0, inx, iny,inz):
    
    outcode = 0   
    outx   = inx * mProj[0][0] + iny * mProj[1][0] + inz * mProj[2][0] +  mProj[3][0]
    outy   = inx * mProj[0][1] + iny * mProj[1][1] + inz * mProj[2][1] +  mProj[3][1] 
    outz   = inx * mProj[0][2] + iny * mProj[1][2] + inz * mProj[2][2] +  mProj[3][2] 
    w      = inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3] 
    
    # global outValues
    # outValues[i*4+0] = outx
    # outValues[i*4+1] = outy
    # outValues[i*4+2] = outz
    # outValues[i*4+3] = w
    
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
    
    
    
    return [raster0, raster1, raster2], originalPixel



# def findEdges(currVetex):
#     flag = 0
#     edge1 = 0
#     edge2 = 0
    
#     for currEdge in edges:
#         if(currVetex == currEdge[0] or currVetex == currEdge[1]):
#             if(flag == 0):
#                 if (currVetex != currEdge[0]):
#                     edge1 = currEdge[0]
#                 else:
#                     edge1 = currEdge[1]
#                 flag = 1
#             else:
#                 if (currVetex != currEdge[0]):
#                     edge2 = currEdge[0]
#                 else:
#                     edge2 = currEdge[1]
#     return edge1, edge2
    

def findEdges(currVetex, edges):
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
    #TODO: we can rewrite this function, but for now let it be, we change it later
    numberOfOccur = 0
    
    for currEdge in edges:
        if(currEdge[0] == outsideVertex or currEdge[1] == outsideVertex):
            numberOfOccur+=1
    
    for i in range(0,numberOfOccur):
        for currEdge in edges:
            if(currEdge[0] == outsideVertex or currEdge[1] == outsideVertex):
                edges.remove(currEdge)
                break

def edgeFunction(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])     
   

def drawTriangle2(pixelCoordinates, currVertexColours ):
    # print("Drawing Triangle: " + str(currTriangle))
    
    
    
    raster0 = pixelCoordinates[0]
    raster1 = pixelCoordinates[1]
    raster2 = pixelCoordinates[2]
    centerPoint = [0,0,0]
    
    centerPoint[0] = (raster0[0]+raster1[0]+raster2[0])/3
    centerPoint[1] = (raster0[1]+raster1[1]+raster2[1])/3
    
    
    
    angle0 = math.atan2(raster0[1] - centerPoint[1],  raster0[0] - centerPoint[0]) * 180 / PI
    angle1 = math.atan2(raster1[1] - centerPoint[1],  raster1[0] - centerPoint[0]) * 180 / PI
    angle2 = math.atan2(raster2[1] - centerPoint[1],  raster2[0] - centerPoint[0]) * 180 / PI
    
    angle0 = angle0 if angle0>0 else (int)(angle0 + 360) % 360
    angle1 = angle1 if angle1>0 else (int)(angle1 + 360) % 360
    angle2 = angle2 if angle2>0 else (int)(angle2 + 360) % 360
    
    minAngle = min(angle0, angle1, angle2)
    
    # print("Center Point: " + str(centerPoint))
    # print("Angle0: " + str(angle0))
    # print("Angle1: " + str(angle1))
    # print("Angle2: " + str(angle2))
    # print("Min Angle: " + str(minAngle))
    
    
    tempV0 = [0,0,0]
    tempV1 = [0,0,0]
    tempV2 = [0,0,0]
    v0flag = 0
    v1flag = 0
    # v0MinDepth = 0
    # v0MaxDepth = 0
    # v1MinDepth = 0
    # v1MaxDepth = 0
    # v2MinDepth = 0
    # v2MaxDepth = 0
       
    
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
    
    # print("xmin", xmin, "xmax", xmax, "ymin", ymin, "ymax", ymax)
    # 
    if (xmin > imageWidth - 1 or xmax < 0 or ymin > imageHeight - 1 or ymax < 0):
        # print("Out of screen")
        return
    
    # print("((((((((((Drawing Triangle)))))))))))))")
    # print("v0Raster: " + str(v0Raster)+ " , v1Raster: " + str(v1Raster)+ " , v2Raster: " + str(v2Raster))
    
    
    x0 = max(0, (int)(math.floor(xmin)))
    x1 = min(imageWidth - 1, (int)(math.floor(xmax)))
    y0 = max(0, (int)(math.floor(ymin)))
    y1 = min(imageHeight - 1, (int)(math.floor(ymax)))
    
    # print("x0", x0, "x1", x1, "y0", y0, "y1", y1)
    
    area = edgeFunction(v0Raster, v1Raster, v2Raster)
    
    # print("area: " + str(area))

    if (area <= 0):
        return

    for y in range(y0, y1+1):
        for x in range(x0, x1+1):
            pixelSample = [x + 0.5, y + 0.5, 0]
            w0 = edgeFunction(v1Raster, v2Raster, pixelSample)
            w1 = edgeFunction(v2Raster, v0Raster, pixelSample)
            w2 = edgeFunction(v0Raster, v1Raster, pixelSample)
           
            if (w0 >= 0 and w1 >= 0 and w2 >= 0):
                w0 = w0 / area
                w1 = w1 / area
                w2 = w2 / area
                oneOverZ = v0Raster[2] * w0 + v1Raster[2] * w1 + v2Raster[2] * w2
                z = 1 / oneOverZ
                z = oneOverZ
                storeZasDepth = z
                z = round(z,10)
                # print("z = ",z)
                
                r = w0 * currVertexColours[0][0]*255 + w1 * currVertexColours[1][0]*255 + w2 * currVertexColours[2][0]*255 
                g = w0 * currVertexColours[0][1]*255 + w1 * currVertexColours[1][1]*255 + w2 * currVertexColours[2][1]*255
                b = w0 * currVertexColours[0][2]*255 + w1 * currVertexColours[1][2]*255 + w2 * currVertexColours[2][2]*255

                r =  currVertexColours[0][0]*255 
                g = currVertexColours[0][1]*255 
                b =  currVertexColours[0][2]*255 
                
                # currMinDepth = v0MinDepth * w0 + v1MinDepth * w1 + v2MinDepth * w2
                # currMaxDepth = v0MaxDepth * w0 + v1MaxDepth * w1 + v2MaxDepth * w2

                # if int(r) != 0 or int(g) != 255 or int(b) != 0 :
                #     print(r, g, b)
                    # exit()
                # r= 0
                # g = 255
                # b =0
               
                if depthBuffer.get(y * imageWidth + x):
                    # print("depthBuffer[y * imageWidth + x] = ",depthBuffer[y * imageWidth + x], "z = ",z)
                    if depthBuffer[y * imageWidth + x] > z:
                        
                        depthBuffer[y * imageWidth + x] = z
                        frameBuffer[y * imageWidth + x] = [int(r),int(g),int(b)]
                else:
                    depthBuffer[y * imageWidth + x] = z
                    frameBuffer[y * imageWidth + x] = [int(r),int(g),int(b)]
               
                # frameBuffer[y * imageWidth + x] = [int(r),int(g),int(b)]
                
                # if currTriangleIntervalImage.get(y*imageWidth+x):
                #     # print(y*imageWidth+x, "already exists")
                #     currValues = currTriangleIntervalImage[y*imageWidth+x]
                #     currValues.append([r,g,b,currMinDepth,currMaxDepth])
                #     currTriangleIntervalImage[y*imageWidth+x] = currValues
                # else:
                #     currTriangleIntervalImage[y*imageWidth+x] = [[r,g,b,currMinDepth,currMaxDepth]]

                # print(y*imageWidth + x,int(r),int(g),int(b))

                

                
    
    
    # comment='any comment string'
    # ftype='P6' #'P6' for binary

    # outputImageFile = open("images/py_" + str(currTriangle) +"_"+str(currImg)+ ".ppm", "w")
    # outputImageFile.write("%s\n" % (ftype))
    # outputImageFile.write("%d %d\n" % (imageWidth, imageHeight)) 
    # outputImageFile.write("255\n")
    # tempString = ""
    # for i in range(0, imageWidth * imageHeight):
    #     if frameBuffer.get(i):
    #         # outputImageFile.write("%c%c%c" % (frameBuffer[i][0],frameBuffer[i][1],frameBuffer[i][2]))
    #         outputImageFile.write("%c%c%c" % (0,255,0))
    #         print(i,frameBuffer[i][0],frameBuffer[i][1],frameBuffer[i][2])
    #     else:
    #         outputImageFile.write("%c%c%c" % (1,25,24))   
    # outputImageFile.close()    
    
    


def cpp_vertexPlaneIntersectionPoint(insideVertex, outsideVertex, insidevertexW, outsideVertexW, intersectionPlane):

    x1 = insideVertex[0]
    y1 = insideVertex[1]
    z1 = insideVertex[2]
    w1 = insidevertexW

    x2 = outsideVertex[0]
    y2 = outsideVertex[1]
    z2 = outsideVertex[2]
    w2 = outsideVertexW
    
    # print("\n x1 = ",x1, "y1 = ",y1, "z1 = ",z1, "w1 = ",w1)
    # print("x2 = ",x2, "y2 = ",y2, "z2 = ",z2, "w2 = ",w2)

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


def generateTriangles2(tr_vertex_coordinates, tr_vertex_ws, tr_num_of_vertices,edges,currTriangle, currImageColours):
    
    
    raster0 = [0,0,0]
    raster1 = [0,0,0]
    raster2 = [0,0,0]
    
    edge1 = edges[0]
    # print("edge1: " + str(edge1))
    # print("tr_vertex_coordinates[edge1[0]]: " + str(tr_vertex_coordinates[edge1[0]]))
    
    t0 = tr_vertex_coordinates[edge1[0]][0]/ tr_vertex_ws[edge1[0]]
    t1 = tr_vertex_coordinates[edge1[0]][1]/ tr_vertex_ws[edge1[0]]
    t2 = tr_vertex_coordinates[edge1[0]][2]/ tr_vertex_ws[edge1[0]]
    
    # d0 = depthInformation[edge1[0]][1]
    # d1 = depthInformation[edge1[0]][2]
    if np.isnan(t0) or np.isnan(t1) or np.isnan(t2):
        print("NAN value found")
        return
    
    raster0[0] = min(imageWidth - 1, (int)((t0 + 1) * 0.5 * imageWidth)) 
    raster0[1] = min(imageHeight - 1,(int)((1 - (t1 + 1) * 0.5) * imageHeight))
    raster0[2] = t2
    
    firstVertex = edge1[0]
    secondVertex = edge1[1]
    thirdVertex = 0
    prviousFirstVertex = firstVertex
    
    
    for currImg in range(0,tr_num_of_vertices-2):
        vertex3, vertex4 = findEdges(secondVertex,edges)
        thirdVertex = 0
        
        if vertex3!=prviousFirstVertex:
            thirdVertex = vertex3
        else:
            thirdVertex = vertex4
    
        # print("firstVertex: " + str(firstVertex)+ " , secondVertex: " + str(secondVertex)+ " , thirdVertex: " + str(thirdVertex))
 
 
        t0 = tr_vertex_coordinates[secondVertex][0]/ tr_vertex_ws[secondVertex]
        t1 = tr_vertex_coordinates[secondVertex][1]/ tr_vertex_ws[secondVertex]
        t2 = tr_vertex_coordinates[secondVertex][2]/ tr_vertex_ws[secondVertex]

        if np.isnan(t0) or np.isnan(t1) or np.isnan(t2):
            print("NAN value found")
            return
        
        raster1[0] = min(imageWidth - 1, (int)((t0 + 1) * 0.5 * imageWidth)) 
        raster1[1] = min(imageHeight - 1,(int)((1 - (t1 + 1) * 0.5) * imageHeight))
        raster1[2] = t2
        
        
        # d2 = depthInformation[secondVertex][1]
        # d3 = depthInformation[secondVertex][2]
        
        t0 = tr_vertex_coordinates[thirdVertex][0]/ tr_vertex_ws[thirdVertex]
        t1 = tr_vertex_coordinates[thirdVertex][1]/ tr_vertex_ws[thirdVertex]
        t2 = tr_vertex_coordinates[thirdVertex][2]/ tr_vertex_ws[thirdVertex]

        if np.isnan(t0) or np.isnan(t1) or np.isnan(t2):
            print("NAN value found")
            return
        
        raster2[0] = min(imageWidth - 1, (int)((t0 + 1) * 0.5 * imageWidth)) 
        raster2[1] = min(imageHeight - 1,(int)((1 - (t1 + 1) * 0.5) * imageHeight))
        raster2[2] = t2
        
       
        # d4 = depthInformation[thirdVertex][1]
        # d5 = depthInformation[thirdVertex][2]

        # print("Draw traingle : ", raster0,raster1,raster2)
        
        
        currVertexColours = [currImageColours[firstVertex], currImageColours[secondVertex], currImageColours[thirdVertex]]
        # print("Vertex Colours: " + str(currVertexColours))
        
        drawTriangle2([raster0,raster1,raster2], currVertexColours)
        # print("Drawing Triangle Done: " + str(currTriangle))

        prviousFirstVertex =secondVertex
        secondVertex = thirdVertex
    
    
    
    
    

def renderATriangle(currTriangle,xp, yp,zp):
    
    # s2 = Solver()
    # set_param('parallel.enable', True)
    # set_option(rational_to_decimal=True)
    # set_option(precision=1000)
    # set_param('parallel.enable', True)
    # s2.set("sat.local_search_threads", 28)
    # s2.set("sat.threads", 28)
    # # s2.set("timeout",2000)    
    # set_option(max_args=10000000, max_lines=1000000, max_depth=10000000, max_visited=1000000)

    # s2.add(simplify(currGroupRegionCons))
    
    
    vertex0 = nvertices[currTriangle*3+0]
    vertex1 = nvertices[currTriangle*3+1]
    vertex2 = nvertices[currTriangle*3+2]
    currTriangleVertices = [vertex0, vertex1,vertex2]
    
    
    # print("vertex0 = ",vertex0, "vertex1 = ",vertex1, "vertex2 = ",vertex2)
    # print(vertices[currTriangleVertices[0]*3+0])
    # print(vertices[currTriangleVertices[0]*3+1])
    # print(vertices[currTriangleVertices[0]*3+2])
    
    # print(vertices[currTriangleVertices[1]*3+0])
    # print(vertices[currTriangleVertices[1]*3+1])
    # print(vertices[currTriangleVertices[1]*3+2])
    
    # print(vertices[currTriangleVertices[2]*3+0])
    # print(vertices[currTriangleVertices[2]*3+1])
    # print(vertices[currTriangleVertices[2]*3+2])
    

    v0Vertex = [vertices[currTriangleVertices[0]*3+0], vertices[currTriangleVertices[0]*3+1],vertices[currTriangleVertices[0]*3+2] ]
    v1Vertex = [vertices[currTriangleVertices[1]*3+0], vertices[currTriangleVertices[1]*3+1],vertices[currTriangleVertices[1]*3+2] ]
    v2Vertex = [vertices[currTriangleVertices[2]*3+0], vertices[currTriangleVertices[2]*3+1],vertices[currTriangleVertices[2]*3+2] ]


    if Counter(v0Vertex) == Counter(v1Vertex) or Counter(v0Vertex) == Counter(v2Vertex) or Counter(v1Vertex) == Counter(v2Vertex):
        # print(v0Vertex, v1Vertex, v2Vertex)
        # print("Error triangle")
        # print(currTriangle)
        return 0
   
    
    
    # numberOfInvRegions = 0  
    
    
    # vertex_plane_cons = [
    #     [And(True),And(True),And(True),And(True),And(True),And(True)],
    #     [And(True),And(True),And(True),And(True),And(True),And(True)],
    #     [And(True),And(True),And(True),And(True),And(True),And(True)],                 
    #                      ]
    
    # print(vertex_plane_cons)
    
   
    # for l in range(0,3):
    #     for m in range(0,6):
    #         # print(l,m)
    #         vertex_plane_cons[l][m] = symComputeOutcodePlane(m, vertices[currTriangleVertices[l]*3+0] -xp0,
    #                                                          vertices[currTriangleVertices[l]*3+1] -yp0,
    #                                                          vertices[currTriangleVertices[l]*3+2]-zp0)
            # print(vertex_plane_cons[0][1])

    # vertex0Inside_cons = And (vertex_plane_cons[0][0] == 0, vertex_plane_cons[0][1] == 0,vertex_plane_cons[0][2] == 0,
    #                         vertex_plane_cons[0][3] == 0,vertex_plane_cons[0][5] == 0,vertex_plane_cons[0][4] == 0)
    
    # vertex1Inside_cons = And (vertex_plane_cons[1][0] == 0, vertex_plane_cons[1][1] == 0,vertex_plane_cons[1][2] == 0,
    #                         vertex_plane_cons[1][3] == 0,vertex_plane_cons[1][5] == 0,vertex_plane_cons[1][4] == 0)
    
    # vertex2Inside_cons = And (vertex_plane_cons[2][0] == 0, vertex_plane_cons[2][1] == 0,vertex_plane_cons[2][2] == 0,
    #                         vertex_plane_cons[2][3] == 0,vertex_plane_cons[2][5] == 0, vertex_plane_cons[2][4] == 0)
    
    # anyOneVertexFullyInside = Or(vertex0Inside_cons,vertex1Inside_cons,vertex2Inside_cons)
    
    
    # print(" tedges[(currTriangle) *6+0] = ",  tedges[(currTriangle) *6+0])
    # print(" tedges[(currTriangle) *6+1] = ",  tedges[(currTriangle) *6+1])
    # print(" tedges[(currTriangle) *6+2] = ",  tedges[(currTriangle) *6+2])
    
    tempList1 = [tedges[(currTriangle) *6+0],tedges[(currTriangle) *6+1],tedges[(currTriangle) *6+2],
                 tedges[(currTriangle) *6+3],tedges[(currTriangle) *6+4],tedges[(currTriangle) *6+5]]
    for el in tempList1 :
        if el not in currTriangleVertices:
            # print("currTriangleVertices not in tempList1")
            # print("edge list = ", tempList1)
            # print("vertex index list = ", currTriangleVertices)
            # print("curent triangle = ", currTriangle)
            # sleep(1)
            errorTriangleList.append(currTriangle)
            return
    
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
    
    
    # print("Triangle edge vertex indices")
    # print(edge0_v0,edge0_v1)
    # print(edge1_v0,edge1_v1)
    # print(edge2_v0,edge2_v1)
    
    
    # fullyOutsideOfPlane_0 = And( vertex_plane_cons[edge0_v0][0] ==1, vertex_plane_cons[edge0_v1][0] ==1,
    #                              vertex_plane_cons[edge1_v0][0] ==1, vertex_plane_cons[edge1_v1][0] ==1,
    #                              vertex_plane_cons[edge2_v0][0] ==1, vertex_plane_cons[edge2_v1][0] ==1)
    
    
    
    # edge0_fullyOutside_cons = Or (
    #         And( vertex_plane_cons[edge0_v0][0] ==1, vertex_plane_cons[edge0_v1][0] ==1),
    #         And( vertex_plane_cons[edge0_v0][1] ==1, vertex_plane_cons[edge0_v1][1] ==1),
    #         And( vertex_plane_cons[edge0_v0][2] ==1, vertex_plane_cons[edge0_v1][2] ==1),
    #         And( vertex_plane_cons[edge0_v0][3] ==1, vertex_plane_cons[edge0_v1][3] ==1), 
    #         And( vertex_plane_cons[edge0_v0][4] ==1, vertex_plane_cons[edge0_v1][4] ==1),
    #         And( vertex_plane_cons[edge0_v0][5] ==1, vertex_plane_cons[edge0_v1][5] ==1)
            
    #         )
    
    # edge1_fullyOutside_cons = Or (
    #             And( vertex_plane_cons[edge1_v0][0] ==1, vertex_plane_cons[edge1_v1][0] ==1),
    #             And( vertex_plane_cons[edge1_v0][1] ==1, vertex_plane_cons[edge1_v1][1] ==1),
    #             And( vertex_plane_cons[edge1_v0][2] ==1, vertex_plane_cons[edge1_v1][2] ==1),
    #             And( vertex_plane_cons[edge1_v0][3] ==1, vertex_plane_cons[edge1_v1][3] ==1), 
    #             And( vertex_plane_cons[edge1_v0][4] ==1, vertex_plane_cons[edge1_v1][4] ==1),
    #             And( vertex_plane_cons[edge1_v0][5] ==1, vertex_plane_cons[edge1_v1][5] ==1)
    # )
    
    # edge2_fullyOutside_cons = Or (
    #             And( vertex_plane_cons[edge2_v0][0] ==1, vertex_plane_cons[edge2_v1][0] ==1),
    #             And( vertex_plane_cons[edge2_v0][1] ==1, vertex_plane_cons[edge2_v1][1] ==1),
    #             And( vertex_plane_cons[edge2_v0][2] ==1, vertex_plane_cons[edge2_v1][2] ==1),
    #             And( vertex_plane_cons[edge2_v0][3] ==1, vertex_plane_cons[edge2_v1][3] ==1), 
    #             And( vertex_plane_cons[edge2_v0][4] ==1, vertex_plane_cons[edge2_v1][4] ==1),
    #             And( vertex_plane_cons[edge2_v0][5] ==1, vertex_plane_cons[edge2_v1][5] ==1),
                
    #             )

    # atleastOneEdgeIntersect_cons = Not(And(edge0_fullyOutside_cons, edge1_fullyOutside_cons, edge2_fullyOutside_cons))
    
    # triangleInsideOutsideCons = Or(anyOneVertexFullyInside, atleastOneEdgeIntersect_cons)
    
    
    # s2.add(simplify(triangleInsideOutsideCons))
    
    # # print(s2.check())
    # print(s2.model())
    
    # s2.push()
    
    # allCamPoses = dict()
   
    
    
    
    # currTriangleInvPositions = [0]*10000*3
    # currTriangleInvDepths = [0]*10000*2
    
    
    

    # globalCurrentImage.clear()
    # globalInsideVertexDataToPPL.clear()
    # globalIntersectingVertexDataToPPL.clear()
    
    # dataToComputeIntervalImage.clear()
    # currTriangleIntervalImage.clear()
    
    # while(1==1):
    # if(s2.check() ==sat):
    # print("\n\n\n==***********************************==========================\n\n")
    # print("starting for new point")
    
    
    currImageColours = dict()
    currImageColours.clear()
    
    currImageColours[0] = [environment.vertColours[currTriangleVertices[0]*3+0],
                            environment.vertColours[currTriangleVertices[0]*3+1],
                            environment.vertColours[currTriangleVertices[0]*3+2]] 
    
    currImageColours[1] = [environment.vertColours[currTriangleVertices[1]*3+0],
                            environment.vertColours[currTriangleVertices[1]*3+1],
                            environment.vertColours[currTriangleVertices[1]*3+2]] 
    
    currImageColours[2] = [environment.vertColours[currTriangleVertices[2]*3+0],
                            environment.vertColours[currTriangleVertices[2]*3+1],
                            environment.vertColours[currTriangleVertices[2]*3+2]] 
    
    
    
    # newvertices = [0]*numOfVertices*3*5
    # newVerticesNumber = 0
    # pixelValueComputed = [0]*numOfVertices*5
    # pixelValues = [0]*numOfVertices*2*5
    
    # edgesInSmallPyramid = [0]*numOftedges
    
    insideVertexDetailsToPPL = [] #store vertex index number,xpixel,ypixel
    numberOfFullyInsideVertices = 0
    numberOfIntersectingEdges = 0
    intersectingEdgeDataToPPL = []
    
    fullyInsideVerticesNumber = []
    intersectingVerticesNumber = []
    
    intersectingData = dict()
    intersectingData.clear()

    # globalCurrentImage.clear()
    # globalInsideVertexDataToPPL.clear()
    # globalIntersectingVertexDataToPPL.clear()
    
    edges.clear()
    
    tr_num_of_vertices = 3
    tr_curr_num_of_vertex = 3
    tr_vertex_coordinates = np.zeros((100,3))
    tr_vertex_ws = [0]*100
    tr_vertex_outcodes = np.zeros((100,6))
    tr_vertices_set = []
    
    
    # print(datetime.now())
    # print(s2.check())
    # m = s2.model()
    # print(m)
    # sleep(1)
    posXp = xp
    posYp = yp
    posZp = zp

    # notTheCurrentPosCons1 = Or(xp0!= m[xp0], yp0!=m[yp0],zp0!= m[zp0])
    # currentPlane = 0 #top plane
    
    # outcodeP0 = [0]*numOfVertices*6
    outcodeP0 = [0]*30*6
        
    # outValue0, outW0 = computeOutcodeAtPos(currTriangleVertices[0],outcodeP0, 
    #                     vertices[currTriangleVertices[0]*3+0]-posXp ,
    #                     vertices[currTriangleVertices[0]*3+1]-posYp,
    #                     vertices[currTriangleVertices[0]*3+2]-posZp )
    # outValue1, outW1 = computeOutcodeAtPos(currTriangleVertices[1],outcodeP0, 
    #                     vertices[currTriangleVertices[1]*3+0]-posXp ,
    #                     vertices[currTriangleVertices[1]*3+1]-posYp,
                        
    #                     vertices[currTriangleVertices[1]*3+2]-posZp )
    
    # outValue2, outW2 = computeOutcodeAtPos(currTriangleVertices[2],outcodeP0, 
    #                     vertices[currTriangleVertices[2]*3+0]-posXp ,
    #                     vertices[currTriangleVertices[2]*3+1]-posYp,
    #                     vertices[currTriangleVertices[2]*3+2]-posZp )
    
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
    
    
    # print("Outcodes = ",outcodeP0)
    # print("outValue0 = ",outValue0, "outValue1 = ",outValue1, "outValue2 = ",outValue2)
    # print("outW0 = ",outW0, "outW1 = ",outW1, "outW2 = ",outW2)
    # print("posXp = ",posXp, "posYp = ",posYp, "posZp = ",posZp)
    
    
    # outValueToWorld0 = outValueToWorldCoordinates([outValue0[0], outValue0[1], outValue0[2]])
    # outValueToWorld1 = outValueToWorldCoordinates([outValue1[0], outValue1[1], outValue1[2]])
    # outValueToWorld2 = outValueToWorldCoordinates([outValue2[0], outValue2[1], outValue2[2]])
    
    # print("outValueToWorldCoordinates0 = ", outValueToWorld0)
    # print("worldvalue0 = ", outValueToWorld0[0]+posXp, outValueToWorld0[1]+posYp, outValueToWorld0[2]+posZp)
    
    # print("outValueToWorldCoordinates1 = ", outValueToWorld1)
    # print("worldvalue1 = ", outValueToWorld1[0]+posXp, outValueToWorld1[1]+posYp, outValueToWorld1[2]+posZp)
    # print("outValueToWorldCoordinates2 = ", outValueToWorld2)
    # print("worldvalue2 = ", outValueToWorld2[0]+posXp, outValueToWorld2[1]+posYp, outValueToWorld2[2]+posZp)
    # /print("mProj = ", mProj)
    
    
    
    bit0 = outcodeP0[0] & outcodeP0[6] & outcodeP0[12]
    bit1 = outcodeP0[1] & outcodeP0[7] & outcodeP0[13]
    bit2 = outcodeP0[2] & outcodeP0[8] & outcodeP0[14]
    bit3 = outcodeP0[3] & outcodeP0[9] & outcodeP0[15]
    bit4 = outcodeP0[4] & outcodeP0[10] & outcodeP0[16]
    bit5 = outcodeP0[5] & outcodeP0[11] & outcodeP0[17]
    
    # print("bit0 ==> ", bit0)
    
    
    tr_vertex_outcodes[0] = outcodeP0[0:6]
    tr_vertex_outcodes[1] = outcodeP0[6:12]
    tr_vertex_outcodes[2] = outcodeP0[12:18]

    tr_vertex_coordinates[0] = outValue0
    tr_vertex_coordinates[1] = outValue1
    tr_vertex_coordinates[2] = outValue2

    tr_vertex_ws[0]=outW0
    tr_vertex_ws[1]=outW1
    tr_vertex_ws[2]=outW2


    tr_vertices_set.append(0)
    tr_vertices_set.append(1)
    tr_vertices_set.append(2)

    edges.append([edge0_v0,edge0_v1])
    edges.append([edge1_v0,edge1_v1])
    edges.append([edge2_v0,edge2_v1])
    
    # print("Edges = ",edges)
    
            
    if(not any(outcodeP0)):
        # print("All vertices inside the frustum")
        # print("Adding data to PPL insideVertexDetailsToPPL")     
        # for currVert in range(0,3):        
            
        #     # pixelValueComputed[edge_v0] = 1
        #     print("\nComputing pixel value of vertex ", currVert)
        #     # x = vertices[currTriangleVertices[currVert]*3+0]
        #     # y = vertices[currTriangleVertices[currVert]*3+1]
        #     # z = vertices[currTriangleVertices[currVert]*3+2]
        #     # vertexPixelValue = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
        #     # print("vertexPixelValue ==> ",x,y,z, vertexPixelValue)
            
            
            
        #     # currVertexPixelData = [] 
        #     # tempPixelData = [currTriangleVertices[currVert], currPixelValue[0],currPixelValue[1],currVert]
        #     # insideVertexDetailsToPPL.append(tempPixelData)
        #     # # currVertexPixelData.append(tempPixelData)  
        #     # globalInsideVertexDataToPPL.append(tempPixelData) 
        #     # numberOfFullyInsideVertices+=1 
        #     # fullyInsideVerticesNumber.append(currVert)  
        #     # currImage.append([ currPixelValue[0],currPixelValue[1]])   

        #     # fflag = 0
        #     # currVertexPixelData = [] 
        #     # for vPixels in range(0,len(vertexPixelValue),2):
        #     #     if(vertexPixelValue[vPixels] >=0 and vertexPixelValue[vPixels+1]>=0):
        #     #         currVertColour = environment.vertColours[currTriangleVertices[currVert]*3:currTriangleVertices[currVert]*3+3]
        #     #         tempPixelData = [currTriangleVertices[currVert], vertexPixelValue[vPixels],vertexPixelValue[vPixels+1],currVert,currVertColour]
        #     #         insideVertexDetailsToPPL.append(tempPixelData)
        #     #         currVertexPixelData.append(tempPixelData)                            
        #     #         if fflag == 0:
        #     #             numberOfFullyInsideVertices+=1
        #     #             fflag = 1
        #     #         currImage.append([vertexPixelValue[vPixels],vertexPixelValue[vPixels+1]])
        #     #         print(" pixelvalue : ", vertexPixelValue)
        #     #         # print("Adding data to PPL insideVertexDetailsToPPL")
        #     # if fflag == 1:
        #     #     globalInsideVertexDataToPPL.append(currVertexPixelData)
        #     #     fullyInsideVerticesNumber.append(currVert)
        
        currPixelValue = {}
        
        currPixelValue[0], temp = pixelValue(outValue0,outW0)        
        currPixelValue[1], temp = pixelValue(outValue1,outW1)        
        currPixelValue[2], temp = pixelValue(outValue2,outW2)
        
        # print("currPixelValue using python ==> ", currPixelValue) 
        # print("draw triangle")
        
        # currPixelValue[0] = [100, 100, 0.6]
        # currPixelValue[1] = [700, 100, 0.6]
        # currPixelValue[2] = [540, 540, 0.6]
        
       
        
        drawTriangle2(currPixelValue, currImageColours)
        
        
        
        
    
    elif(bit0 or bit1 or bit2 or bit3 or bit4 or bit5):            
        # print("All vertices outside the frustum, covered already, no need to add to check")
        pass
    else:
        # print("Some vertices inside and some outside the frustum. Intersecting edges, clipping needed")
                
        
        #find data of the fully inside vertices
        # vert0_pos = outcodeP0[0] | outcodeP0[1] | outcodeP0[2] | outcodeP0[3] | outcodeP0[4] | outcodeP0[5]
        # vert1_pos = outcodeP0[6] | outcodeP0[7] | outcodeP0[8] | outcodeP0[9] | outcodeP0[10] | outcodeP0[11]
        # vert2_pos = outcodeP0[12] | outcodeP0[13] | outcodeP0[14] | outcodeP0[15] | outcodeP0[16] | outcodeP0[17]
        
        
        # if(not vert0_pos):
        #     currVert =0
        #     print("First vertex is inside the frustum")
        #     print("\nComputing pixel value of vertex ", currVert)
            
        #     # currPixelValue = []
        #     # currPixelValue, temp = pixelValue(outValue0,outW0)              
        #     # print("currPixelValue using python ==> ", currPixelValue) 
        #     # currVertexPixelData = []
        #     # tempPixelData = [currTriangleVertices[currVert], currPixelValue[0],currPixelValue[1],currVert]
        #     # insideVertexDetailsToPPL.append(tempPixelData)
        #     # currVertexPixelData.append(tempPixelData)  
        #     # globalInsideVertexDataToPPL.append(tempPixelData) 
        #     # numberOfFullyInsideVertices+=1 
            
        #     # fullyInsideVerticesNumber.append(currVert)   
        #     # currImage.append([ currPixelValue[0],currPixelValue[1]])  
            
            
            
            
        #     x = vertices[currTriangleVertices[currVert]*3+0]
        #     y = vertices[currTriangleVertices[currVert]*3+1]
        #     z = vertices[currTriangleVertices[currVert]*3+2]
        #     vertexPixelValue = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
        #     print("vertexPixelValue ==> ",x,y,z, vertexPixelValue)

        #     fflag = 0
        #     currVertexPixelData = [] 
        #     for vPixels in range(0,len(vertexPixelValue),2):
        #         if(vertexPixelValue[vPixels] >=0 and vertexPixelValue[vPixels+1]>=0):
        #             currVertColour = environment.vertColours[currTriangleVertices[currVert]*3:currTriangleVertices[currVert]*3+3]
        #             tempPixelData = [currTriangleVertices[currVert], vertexPixelValue[vPixels],vertexPixelValue[vPixels+1],currVert,currVertColour]
        #             insideVertexDetailsToPPL.append(tempPixelData)
        #             currVertexPixelData.append(tempPixelData)                            
        #             if fflag == 0:
        #                 numberOfFullyInsideVertices+=1
        #                 fflag = 1
        #             currImage.append([vertexPixelValue[vPixels],vertexPixelValue[vPixels+1]])
        #             # print("edge_v0: ", edge_v0," pixelvalue : ", vertexPixelValue)
        #             # print("Adding data to PPL insideVertexDetailsToPPL")
        #     if fflag == 1:
        #         globalInsideVertexDataToPPL.append(currVertexPixelData)
        #         fullyInsideVerticesNumber.append(currVert)
            
        
        # if(not vert1_pos):
        #     currVert =1
        #     print("Second vertex is inside the frustum")
        #     print("\nComputing pixel value of vertex ", currVert)
            
        #     currPixelValue = []
        #     currPixelValue, temp = pixelValue(outValue1,outW1)              
        #     print("currPixelValue using python ==> ", currPixelValue) 
        #     # currVertexPixelData = [] 
        #     # tempPixelData = [currTriangleVertices[currVert], currPixelValue[0],currPixelValue[1],currVert]
        #     # insideVertexDetailsToPPL.append(tempPixelData)
        #     # currVertexPixelData.append(tempPixelData)  
        #     # globalInsideVertexDataToPPL.append(tempPixelData) 
        #     # numberOfFullyInsideVertices+=1 
        #     # fullyInsideVerticesNumber.append(currVert)     
        #     # currImage.append([ currPixelValue[0],currPixelValue[1]])  
            
            
        #     x = vertices[currTriangleVertices[currVert]*3+0]
        #     y = vertices[currTriangleVertices[currVert]*3+1]
        #     z = vertices[currTriangleVertices[currVert]*3+2]
        #     vertexPixelValue = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
        #     print("vertexPixelValue ==> ",x,y,z, vertexPixelValue)

        #     fflag = 0
        #     currVertexPixelData = [] 
        #     for vPixels in range(0,len(vertexPixelValue),2):
        #         if(vertexPixelValue[vPixels] >=0 and vertexPixelValue[vPixels+1]>=0):
        #             currVertColour = environment.vertColours[currTriangleVertices[currVert]*3:currTriangleVertices[currVert]*3+3]
        #             tempPixelData = [currTriangleVertices[currVert], vertexPixelValue[vPixels],vertexPixelValue[vPixels+1],currVert, currVertColour]
        #             insideVertexDetailsToPPL.append(tempPixelData)
        #             currVertexPixelData.append(tempPixelData)                            
        #             if fflag == 0:
        #                 numberOfFullyInsideVertices+=1
        #                 fflag = 1
        #             currImage.append([vertexPixelValue[vPixels],vertexPixelValue[vPixels+1]])
        #             # print("edge_v0: ", edge_v0," pixelvalue : ", vertexPixelValue)
        #             # print("Adding data to PPL insideVertexDetailsToPPL")
        #     if fflag == 1:
        #         globalInsideVertexDataToPPL.append(currVertexPixelData)
        #         fullyInsideVerticesNumber.append(currVert)
        
        # if(not vert2_pos):
        #     currVert =2
        #     print("Third vertex is inside the frustum")
        #     print("\nComputing pixel value of vertex ", currVert)
            
            
        #     # currPixelValue = []
        #     # currPixelValue, temp = pixelValue(outValue2,outW2)              
        #     # print("currPixelValue using python ==> ", currPixelValue) 
        #     # currVertexPixelData =[]
        #     # tempPixelData = [currTriangleVertices[currVert], currPixelValue[0],currPixelValue[1],currVert]
        #     # insideVertexDetailsToPPL.append(tempPixelData)
        #     # currVertexPixelData.append(tempPixelData)  
        #     # globalInsideVertexDataToPPL.append(tempPixelData) 
        #     # numberOfFullyInsideVertices+=1    
        #     # fullyInsideVerticesNumber.append(currVert)
        #     # currImage.append([ currPixelValue[0],currPixelValue[1]])  
            
        #     x = vertices[currTriangleVertices[currVert]*3+0]
        #     y = vertices[currTriangleVertices[currVert]*3+1]
        #     z = vertices[currTriangleVertices[currVert]*3+2]
        #     vertexPixelValue = getVertexPixelValueZ3(m[xp0],m[yp0],m[zp0],x,y,z)
        #     print("vertexPixelValue ==> ",x,y,z, vertexPixelValue)

        #     fflag = 0
        #     currVertexPixelData = [] 
        #     for vPixels in range(0,len(vertexPixelValue),2):
        #         if(vertexPixelValue[vPixels] >=0 and vertexPixelValue[vPixels+1]>=0):
        #             currVertColour = environment.vertColours[currTriangleVertices[currVert]*3:currTriangleVertices[currVert]*3+3]
        #             tempPixelData = [currTriangleVertices[currVert], vertexPixelValue[vPixels],vertexPixelValue[vPixels+1],currVert, currVertColour]
        #             insideVertexDetailsToPPL.append(tempPixelData)
        #             currVertexPixelData.append(tempPixelData)                            
        #             if fflag == 0:
        #                 numberOfFullyInsideVertices+=1
        #                 fflag = 1
        #             currImage.append([vertexPixelValue[vPixels],vertexPixelValue[vPixels+1]])
        #             # print("edge_v0: ", edge_v0," pixelvalue : ", vertexPixelValue)
        #             # print("Adding data to PPL insideVertexDetailsToPPL")
        #     if fflag == 1:
        #         globalInsideVertexDataToPPL.append(currVertexPixelData)
        #         fullyInsideVerticesNumber.append(currVert)
        
        
        # print("\nStart Clipping")
        
        


    

            

        num_of_planes = 6
        
        # print("tr_vertices_set = ", tr_vertices_set)
        # print("tr_vertex_outcodes = ", tr_vertex_outcodes)
        # print("edges = ", edges)
        # for currPlane in range(0, num_of_planes):
        for currPlane in [5,0,1,2,3, ]:
            # print("\n\n========================\n")
            # print("current Plane = ", currPlane )
            
            tr_outside_vertex_set = []
            num_of_outside_vertices = 0
            
            for currVert in tr_vertices_set:
                if(tr_vertex_outcodes[currVert][currPlane] == 1):
                    # print( " vertex outside the plane  is ", currVert)
                    num_of_outside_vertices +=1
                    tr_outside_vertex_set.append(currVert)
            
            # print("current number of outside vertices = ", num_of_outside_vertices)
            
            while(num_of_outside_vertices > 2):
                for currOutsideVert in tr_outside_vertex_set:               
                    edge1, edge2 = findEdges(currOutsideVert,edges)
                    # print("edge1 = ", edge1, "edge2 = ", edge2)
                    if(tr_outside_vertex_set.count(edge1) and tr_outside_vertex_set.count(edge2)):
                        # print("the vertex ",currOutsideVert ," is completely outside")
                        # print("removing the edge ", edge1, edge2)
                        removeEdges(currOutsideVert)
                        tr_vertices_set.remove(currOutsideVert)
                        tr_outside_vertex_set.remove(currOutsideVert)
                        tr_num_of_vertices = tr_num_of_vertices-1
                        edges.append([edge1,edge2])
                        break
                num_of_outside_vertices -=1
            
            if num_of_outside_vertices == 1:
                
                
                outsideVertex = tr_outside_vertex_set[0]
                # print("outsideVertex = ", outsideVertex)
                edge1,edge2 = findEdges(outsideVertex, edges)
                # print("edge1 = ", edge1, "edge2 = ", edge2)
                
                

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
                
                # print("intersectionPoint1 = ", intersectionPoint1)
                # print("intersectionPoint2 = ", intersectionPoint2)
                # print("intersectionPoint1W = ", intersectionPoint1W)
                # print("intersectionPoint2W = ", intersectionPoint2W)
                
                # print("pixel values of intersection points \n---------")
                # print("intersectionPoint1 = ", pixelValue(intersectionPoint1, intersectionPoint1W))
                # print("intersectionPoint2 = ", pixelValue(intersectionPoint2, intersectionPoint2W))
                
                
                
                if currPlane == 5:
                    # intersectionPoint1[0] = -intersectionPoint1[0] 
                    # intersectionPoint1[1] = -intersectionPoint1[1]
                    
                    intersectionPoint1W = -intersectionPoint1W
                    
                    # intersectionPoint2[0] = -intersectionPoint2[0]
                    # intersectionPoint2[1] = -intersectionPoint2[1]
                    
                    intersectionPoint2W = -intersectionPoint2W
                
                
                # print("pixel values of intersection pointsAfter \n---------")
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
                
               
                
                edge1_1, edge1_2 =  findEdges(outsideVertex1, edges)

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
                
                
                # print("Edges = ",edges)
                # print("edge1_1 = ", edge1_1, "edge1_2 = ", edge1_2)
                # print("insidevertexOfoutside1 = ", insidevertexOfoutside1, "outsidevertexOfOutside1 = ", outsidevertexOfOutside1)
                
                
                
                insideVertex1_cord = tr_vertex_coordinates[insidevertexOfoutside1]
                insideVertex1W = tr_vertex_ws[insidevertexOfoutside1]                    
                outsideVertex1_cord = tr_vertex_coordinates[outsideVertex1]
                outsideVertex1W =tr_vertex_ws[outsideVertex1]
                
                prop_t1, intersectionPoint1, intersectionPoint1W = cpp_vertexPlaneIntersectionPoint(insideVertex1_cord, outsideVertex1_cord, 
                                                                                    insideVertex1W, outsideVertex1W, currPlane)

                removeEdges(outsideVertex1)
        
                tr_vertices_set.remove(outsideVertex1)
                tr_num_of_vertices = tr_num_of_vertices-1
                
                # print("intersectionPoint1 = ", intersectionPoint1)
                # print("intersectionPoint1W = ", intersectionPoint1W)
                
                if currPlane == 5:
                    # intersectionPoint1[0] = -intersectionPoint1[0]
                    # intersectionPoint1[1] = -intersectionPoint1[1]
                    # intersectionPoint1[2] = intersectionPoint1[2]                
                    intersectionPoint1W = - intersectionPoint1W
                    
                    
                # print("intersectionPoint1 = ", intersectionPoint1)
                # print("intersectionPoint1W = ", intersectionPoint1W )
                
                
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

                
                
                edge2_1, edge2_2 =  findEdges(outsideVertex2, edges)
                
                
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
                                 
                    intersectionPoint2W = - intersectionPoint2W
                
                
                
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
                
                
                
                


                # print("pixel values of intersection points \n---------")
                # print("intersectionPoint1 = ", pixelValue(intersectionPoint1, intersectionPoint1W))
                # print("intersectionPoint2 = ", pixelValue(intersectionPoint2, intersectionPoint2W))
        
        generateTriangles2(tr_vertex_coordinates, tr_vertex_ws, tr_num_of_vertices,edges,currTriangle, currImageColours)
        
        return

        # exit()
        
        # print("\n=========statistics1=========")
        # print("Vertices after the clipping operation = ", tr_vertices_set)
        # print("number of vertices after clipping = ", tr_num_of_vertices)
        # print("edges after clipping = ", edges) 
        # print("number of edges after clipping = ", len(edges))
        # # print("vertex coordinates after clipping = ", tr_vertex_coordinates)
        # print("vertex ws after clipping = ", tr_vertex_ws)
        # print("Fully inside vertices = ", fullyInsideVerticesNumber)
        # print("Intersecting data = ", intersectingData)
        # print("\n==============================")     
                   

        # intersectingVertices = list(set(tr_vertices_set) - set(fullyInsideVerticesNumber))
        # print("Intersecting vertices to consider = ",intersectingVertices)
        
        # #DOING
        # #compute world coordinates of the end points of each edge
        # #compute pixel coordinates of the intersecting point
        # #prepare the intersecting data to pass to PPL
        
        # for currVert in intersectingVertices:
        #     print("\ncurrent vertex to consider = ", currVert)
        #     currIntersectingData = intersectingData[currVert]
        #     print("currIntersectingData = ", currIntersectingData)
            
        #     currInsideVertCoordinates =[0,0,0]
        #     currOutsideVertCoordinates =[0,0,0]
            
        #     if currIntersectingData[1] <3:
        #         currInsideVertCoordinates = vertices[currTriangleVertices[currIntersectingData[1]]*3:currTriangleVertices[currIntersectingData[1]]*3+3]
        #     else:   
        #         tempCoordinates = outValueToWorldCoordinates(tr_vertex_coordinates[currIntersectingData[1]])
        #         currInsideVertCoordinates = [tempCoordinates[0]+posXp, tempCoordinates[1]+posYp, tempCoordinates[2]+posZp]
            
        #     if currIntersectingData[2] <3:
        #         currOutsideVertCoordinates = vertices[currTriangleVertices[currIntersectingData[2]]*3:currTriangleVertices[currIntersectingData[2]]*3+3]
        #     else:
        #         tempCoordinates = outValueToWorldCoordinates(tr_vertex_coordinates[currIntersectingData[2]])
        #         currOutsideVertCoordinates = [tempCoordinates[0]+posXp, tempCoordinates[1]+posYp, tempCoordinates[2]+posZp]
        
        #     currentIntersectingPlane = currIntersectingData[3]
            
        #     isIntersect, vertexPixelValue2,intersectionPoint,mp,mq =planeEdgeIntersectionUpdated(currentIntersectingPlane,currInsideVertCoordinates, currOutsideVertCoordinates,m,1)
                   
            
        #     # print("currInsideVertCoordinates = ", currInsideVertCoordinates)
        #     # print("currOutsideVertCoordinates = ", currOutsideVertCoordinates)
        #     # print("currentIntersectingPlane = ", currentIntersectingPlane)
        #     # print("isIntersect = ", isIntersect)
        #     # print("vertexPixelValue2 = ", vertexPixelValue2)
        #     # print("intersectionPoint = ", intersectionPoint)
        #     # print("mp = ", mp)
        #     # print("mq = ", mq)
            
        #     if( isIntersect== 1):
        #         # currentIntersectingPlane =0
        #         x = intersectionPoint[0]
        #         y = intersectionPoint[1]
        #         z = intersectionPoint[2]

        #         fflag =0
        #         currIntersectionData = []
        #         for vPixel in range(0, len(vertexPixelValue2),2):
        #             if( (vertexPixelValue2[vPixel]>=0 and vertexPixelValue2[vPixel]<=49) and 
        #                 (vertexPixelValue2[vPixel+1]>=0 and vertexPixelValue2[vPixel+1]<=49)
        #                 ) : 
                        
        #                 if fflag == 0:
        #                     # numberOfIntersectingPlanes +=1
        #                     numberOfIntersectingEdges += 1
        #                     fflag = 1
                        
        #                 xpixel = vertexPixelValue2[vPixel]
        #                 ypixel = vertexPixelValue2[vPixel+1]    
                        
        #                 if((currIntersectingData[1] >2 and currIntersectingData[2] >2) and ((xpixel == 0 and ypixel ==0) or(xpixel ==49 and ypixel==0) or
        #                                                                                     (xpixel == 0 and ypixel ==49) or(xpixel ==49 and ypixel==49))):
                            
        #                     singleIntersectingData = [-2, currInsideVertCoordinates,currOutsideVertCoordinates,currentIntersectingPlane,\
        #                                         xpixel,ypixel, currIntersectingData[1], currIntersectingData[2], currIntersectingData[0]] 
        #                 else:                                                      
        #                     singleIntersectingData = [-1, currInsideVertCoordinates,currOutsideVertCoordinates,currentIntersectingPlane,\
        #                                         xpixel,ypixel, currIntersectingData[1], currIntersectingData[2], currIntersectingData[0]]                            
        #                 intersectingEdgeDataToPPL.append(singleIntersectingData) 
        #                 currIntersectionData.append(singleIntersectingData)                           
        #                 currImage.append([xpixel,ypixel])
        #         if fflag == 1:
        #             globalIntersectingVertexDataToPPL.append(currIntersectionData)
        #     else:
        #         t0 = int(tr_vertex_coordinates[currIntersectingData[0]][0]/tr_vertex_ws[currIntersectingData[0]])
        #         t1 = int(tr_vertex_coordinates[currIntersectingData[0]][1]/tr_vertex_ws[currIntersectingData[0]])
                
        #         xpixel =  min(imageWidth-1, int((t0 + 1) * 0.5 * imageWidth))
        #         ypixel =  min(imageHeight-1, int((1 - (t1 + 1) * 0.5) * imageHeight))
                
            
        #         currIntersectionData = []
                
                
        #         singleIntersectingData = [-3, currInsideVertCoordinates,currOutsideVertCoordinates,currentIntersectingPlane,\
        #                                         xpixel,ypixel, currIntersectingData[1], currIntersectingData[2], currIntersectingData[0]]
                
        #         intersectingEdgeDataToPPL.append(singleIntersectingData) 
        #         currIntersectionData.append(singleIntersectingData)                           
        #         currImage.append([xpixel,ypixel])
        #         numberOfIntersectingEdges += 1
        #         globalIntersectingVertexDataToPPL.append(currIntersectionData)
                
    
    
        
            
        # currImageName = currGroupName+str(numberOfInvRegions)
        # print("currImageName :",currImageName)
        # print("PPL Inv region generation :start")
        
        # ##region computation edited for 27/40####
        
        
        
                    
                    
                  
        # print("\n===================\nnumber of fully inside vertices" , numberOfFullyInsideVertices)
        # print("global inside vertex data to ppl = ", globalInsideVertexDataToPPL)
        # for l in itertools.product(*globalInsideVertexDataToPPL):
        #     print(l)
        

        # print("number of intersecting data = ", numberOfIntersectingEdges)
        # print("globalIntersectingVertexDataToPPL = ", globalIntersectingVertexDataToPPL)
        # for l in itertools.product(*globalIntersectingVertexDataToPPL):
        #     print(l)

        
        # combinedDataToPPL = globalInsideVertexDataToPPL+ globalIntersectingVertexDataToPPL
        

        # # print("computing pixel values: done ")
        # allImages.append(currImage)
        # # print("Passing values to PPL for polyhedron computation") 
        
      
        
        
        
        # print("Number of invariant regions = ", numberOfInvRegions)
        # print("\n combinedDataToPPL = ",combinedDataToPPL)
        # findARegion =0
        # tempRegionCons = []
        # tempConsString = ""
        # dataToComputeDepth = []
        
        # for l in itertools.product(*combinedDataToPPL):
        #     print("\n=========\n",l)
        #     dataToComputeDepth = l
        #     print("Fully inside data")
        #     print(l[0:numberOfFullyInsideVertices])
        #     print("\n\n intersecting data")
        #     print(l[numberOfFullyInsideVertices:])

        #     insideVertexDetailsToPPL = l[0:numberOfFullyInsideVertices]
        #     intersectingEdgeDataToPPL = l[numberOfFullyInsideVertices:]

        #     print("\n", insideVertexDetailsToPPL)
        #     print("\n\n", intersectingEdgeDataToPPL)

        #     currImageSetConStringPolyhedra = pyparma_posInvRegion40.computeRegion(currGroupName,posZp,numberOfFullyInsideVertices,insideVertexDetailsToPPL,\
        #     numberOfIntersectingEdges,intersectingEdgeDataToPPL,posXp,posYp,posZp,m[xp0],m[yp0],m[zp0], outcodeP0,currImageName)
        
        #     print(currImageSetConStringPolyhedra.minimized_constraints())
            
        #     # print("inv region generated")
        #     currImageSetConString = str(currImageSetConStringPolyhedra.minimized_constraints())
        #     currImageSetConString = currImageSetConString.replace("x0","xp0")
        #     currImageSetConString = currImageSetConString.replace("x1","yp0")
        #     currImageSetConString = currImageSetConString.replace("x2","zp0")
        #     currImageSetConString = currImageSetConString.replace(" = ","==")
        #     currImageSetConString = currImageSetConString.replace("Constraint_System {"," ")
        #     currImageSetConString = currImageSetConString.replace("}"," ")
            
        #     if(str(currImageSetConString).replace(" ","") == "-1==0" or str(currImageSetConString).replace(" ","") == "0==-1"):
        #         print("Continue with the next point")
        #         sleep(4)
        #         continue
            
            
            
            
        #     tempConsString = currImageSetConString
        #     findARegion = 1
        #     currImageSetConString = "And("+str(currImageSetConString)+")"
        #     currImageSetCons = eval(currImageSetConString)
            
        #     print("checking pos inclusion")
        #     scheck2 = Solver()
        #     scheck2.add(currImageSetCons)
        #     scheck2.add(currGroupRegionCons)
        #     # # scheck.add(z3invRegionCons)
            
        #     # scheck.push()
        #     scheck2.add(And(xp0 ==m[xp0], yp0 == m[yp0], zp0 == m[zp0]))
        #     if(scheck2.check() != sat):
        #         print("Point Outside the region, Continue with the next point")
        #         sleep(3)
        #         continue
            
             
            
            
            
        #     print("Pos is inside the region")
        #     s2.add(Not(currImageSetCons))             
        #     tempRegionCons.append(currImageSetConString)             
        #     break    
            
       
        # if findARegion == 0:
        #     print("Check for error: -1==0")
        #     criticalFile=open("ErrorLog.txt","a")
            
        #     criticalFile.write("Current Triangle = "+str(currTriangle))
        #     criticalFile.write("currGroupRegionCons = "+str(currGroupRegionCons))
            
        
        #     criticalFile.close()
        #     # exit()
        #     return numberOfInvRegions-1
        
        # currImageSetConString = tempConsString
        
        
        
        # #######################Depth code to copy##################
        # #########################################################
        # #########################################################
        # ### New implementation of depth computation. ##############
        # ########################################################
        # #######################################################
        
        
        # print("\n\n Data to compute the depth info ")
        # print(dataToComputeDepth)  
        
        # print("Fully inside data")
        # print(dataToComputeDepth[0:numberOfFullyInsideVertices])
        # print("\n\n intersecting data")
        # print(dataToComputeDepth[numberOfFullyInsideVertices:])   
        
        
        # depthInformation = dict()
        # depthInformation.clear()
        # for inVert in range(0,numberOfFullyInsideVertices):
        #     vert_x = vertices[dataToComputeDepth[inVert][0]*3+0]
        #     vert_y = vertices[dataToComputeDepth[inVert][0]*3+1]
        #     vert_z = vertices[dataToComputeDepth[inVert][0]*3+2]
        #     print("Computing depth of the fully inside vertex ", inVert)
        #     print(dataToComputeDepth[inVert])
        #     print("Vertex data = ",vert_x,vert_y,vert_z)
            
        #     mindepth = 0
        #     maxdepth = 1000000
        #     # print("finding mindepth")
        #     mindepth = gurobiGetDepths4.getDepthInterval(currImageSetConString,vert_x,vert_y,vert_z, currGroupRegionCons )
        #     mindepth = math.sqrt(mindepth)
        #     # print("mindepth = ", mindepth)
        #     # sleep(3)
        #     # print("finding maxdepth")
        #     # maxdepth = gurobiGetDepths4.getDepthInterval2(currImageSetConString,vert_x,vert_y,vert_z,currGroupRegionCons )
        #     # maxdepth = math.sqrt(maxdepth)
            
        #     maxdepth = mindepth + environment.depthOfTheInitialCube
        #     print("mindepth, maxdepth =", mindepth,maxdepth)
            
        #     depthInformation[dataToComputeDepth[inVert][3]] = [inVert, mindepth,maxdepth]
        # print("\n-------")
        # for intVert in range(0,numberOfIntersectingEdges):
        #     print("Computing depth of the intersecting point ", intVert)
        #     print(dataToComputeDepth[numberOfFullyInsideVertices+intVert])
            
            
        #     mindepth = gurobiGetDepths4.getDepthIntervals3(currImageSetConString,dataToComputeDepth[numberOfFullyInsideVertices+intVert],\
        #                                                             currGroupRegionCons,edgeVertexIndices, currTriangleVertices)
                                                                    
            
        #     # # maxdepth = gurobiGetDepths4.getDepthIntervals4(currImageSetConString,dataToComputeDepth[numberOfFullyInsideVertices+intVert],\
        #     #                                                         currGroupRegionCons,edgeVertexIndices, currTriangleVertices)
                                                                    
        #     mindepth = math.sqrt(mindepth)                                                        
        #     maxdepth = mindepth + environment.depthOfTheInitialCube
            
        #     print("Final mindepth = ", mindepth) 
        #     print("\n\n") 
        #     # sleep(20)          
            
        #     depthInformation[dataToComputeDepth[numberOfFullyInsideVertices+intVert][8]] = [numberOfFullyInsideVertices+intVert,
        #                              mindepth,maxdepth]                                          
                                                                    
        # # exit()                     
        
        
        # ###################depth code ends here ######################
        
        # # print("Current Triangle Interval Image = ", currTriangleIntervalImage)
        # intervalImageFunctions1.computeSingleImage(currImageName, numberOfInvRegions, dataToComputeDepth, depthInformation, 
        #     tr_vertices_set, tr_vertex_coordinates, tr_vertex_ws, edges, numberOfFullyInsideVertices, numberOfIntersectingEdges, tr_num_of_vertices, 
        #     currTriangle, currTriangleIntervalImage, currImageColours)
        # # print("Current Triangle Interval Image = ", currTriangleIntervalImage)
        
        
def computeScreenCoordinates():
    filmAspectRatio = filmApertureWidth / filmApertureHeight
    deviceAspectRatio = imageWidth / imageHeight

    top = ((filmApertureHeight * inchToMm / 2) / focalLength) * nearClippingPlane
    right = ((filmApertureWidth * inchToMm / 2) / focalLength) *  nearClippingPlane
    
    xscale = 1
    yscale = 1
    
    # case kOverscan:
    if (filmAspectRatio > deviceAspectRatio):
        yscale = filmAspectRatio / deviceAspectRatio
    else:
        xscale = deviceAspectRatio / filmAspectRatio;
         
    right *= xscale;
    top *= yscale;
    
    bottom = -top;
    left = -right;
    
    t = top
    r = right
    b = bottom
    l = left
        
    n = nearClippingPlane
    f = farClippingPlane
    global mProj   
    mProj = [
        [2 * n / (r - l), 0, 0, 0],
        [0,2 * n / (t - b),0,0],
        [(r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n), -1 ],
        [0,0,-2 * f * n / (f - n),0]
    ]
    



def renderAnImage(xp, yp, zp, currImage): 
    frameBuffer.clear()
    depthBuffer.clear()  
    computeScreenCoordinates()    
    
    for i in range(0,environment.numOfTriangles):
        renderATriangle(i,xp,yp,zp)


    image = array.array('B', [1, 25, 24] * imageWidth * imageHeight)   
    maxval = 255
    for i in range(0, imageWidth * imageHeight):
        if frameBuffer.get(i):
            # print(i)
            # image[i * 3 + 0] = frameBuffer[i][0]
            # image[i * 3 + 1] = frameBuffer[i][1]
            # image[i * 3 + 2] = frameBuffer[i][2]
            image[i * 3 + 0] = max(0, min(255, abs(frameBuffer[i][0])))
            image[i * 3 + 1] = max(0, min(255, abs(frameBuffer[i][1])))
            image[i * 3 + 2] = max(0, min(255, abs(frameBuffer[i][2])))
    ppm_header = f'P6 {imageWidth} {imageHeight} {maxval}\n' 

    # with open("images/py_" + str(currTriangle) +"_"+str(currImage)+ ".ppm", 'wb') as f:   
    with open(str(currImage)+".ppm", 'wb') as f:
        f.write(bytearray(ppm_header, 'ascii'))
        image.tofile(f)
    
    print("Image ", str(currImage)+".ppm saved successfully.")
    
    tempFile =  open("defaultPPM.txt","w")
    for i in range(0, imageWidth * imageHeight):
        if frameBuffer.get(i):
            # print(i)
            tempFile.write(str(max(0, min(255, abs(frameBuffer[i][0]))))+str("\n"))
            tempFile.write(str(max(0, min(255, abs(frameBuffer[i][1]))))+str("\n"))
            tempFile.write(str(max(0, min(255, abs(frameBuffer[i][2]))))+str("\n"))
        else:
            tempFile.write(str(1)+str("\n"))
            tempFile.write(str(25)+str("\n"))
            tempFile.write(str(24)+str("\n"))


# Defining main function
def main():
    numOfArgs = len(sys.argv)
    xPos = float(sys.argv[1])
    yPos = float(sys.argv[2])
    zPos = float(sys.argv[3])

    imgName = sys.argv[4]

    print("Rendering image at : ", xPos, yPos, zPos)
    renderAnImage(xPos, yPos, zPos, imgName)

if __name__=="__main__":
    main()




