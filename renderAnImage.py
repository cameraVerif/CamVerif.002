import scene
import camera

import numpy as np
import math
import array

from collections import Counter
from time import sleep
import sys

frameBuffer = dict()
depthBuffer = dict()

vertices = scene.vertices
nvertices = scene.nvertices
tedges = scene.tedges
vertColours = scene.vertColours

numOfTriangles = scene.numOfTriangles
numOfVertices = scene.numOfVertices
numOfEdges = scene.numOfEdges

imageWidth = camera.imageWidth
imageHeight = camera.imageHeight
canvasWidth = camera.filmApertureWidth
canvasHeight = camera.filmApertureHeight
focalLength = camera.focalLength

PI = 3.14159265358979323846
inchToMm = 25.4
filmApertureWidth = camera.filmApertureWidth
filmApertureHeight = camera.filmApertureHeight

nearClippingPlane = camera.nearClippingPlane
farClippingPlane = camera.farClippingPlane

# # OpenGL perspective projection matrix
mProj = [
        [0, 0, 0, 0],
        [0,0,0,0],
        [0, 0, 0, -1 ],
        [0,0,0,0]
    ]

edges = []
    
def computeOutcodeAtPos(i,outcodeP0, inx, iny,inz):
    
    outcode = 0   
    outx   = inx * mProj[0][0] + iny * mProj[1][0] + inz * mProj[2][0] +  mProj[3][0]
    outy   = inx * mProj[0][1] + iny * mProj[1][1] + inz * mProj[2][1] +  mProj[3][1] 
    outz   = inx * mProj[0][2] + iny * mProj[1][2] + inz * mProj[2][2] +  mProj[3][2] 
    w      = inx * mProj[0][3] + iny * mProj[1][3] + inz * mProj[2][3] +  mProj[3][3] 
  
    outValueToReturn = [outx, outy, outz]
   
    if( not(-w <= outx) ):
        outcodeP0[i*6+3] =1
       
    if(not(outx <=w) ):
        outcodeP0[i*6+2] =1
    
    if( not(-w <= outy) ):
        outcodeP0[i*6+1] =1

    if(not(outy <=w) ):
        outcodeP0[i*6+0] =1
		
    
    if( not(-w <= outz) ):
        outcodeP0[i*6+5] =1

    if(not(outz <=w) ):
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
   
    tempV0 = [0,0,0]
    tempV1 = [0,0,0]
    tempV2 = [0,0,0]
    v0flag = 0
    v1flag = 0
   
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
        # print("Out of screen")
        return
    
    x0 = max(0, (int)(math.floor(xmin)))
    x1 = min(imageWidth - 1, (int)(math.floor(xmax)))
    y0 = max(0, (int)(math.floor(ymin)))
    y1 = min(imageHeight - 1, (int)(math.floor(ymax)))
    
    area = edgeFunction(v0Raster, v1Raster, v2Raster)
   
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
                
                r = w0 * currVertexColours[0][0]*255 + w1 * currVertexColours[1][0]*255 + w2 * currVertexColours[2][0]*255 
                g = w0 * currVertexColours[0][1]*255 + w1 * currVertexColours[1][1]*255 + w2 * currVertexColours[2][1]*255
                b = w0 * currVertexColours[0][2]*255 + w1 * currVertexColours[1][2]*255 + w2 * currVertexColours[2][2]*255

                r =  currVertexColours[0][0]*255 
                g =  currVertexColours[0][1]*255 
                b =  currVertexColours[0][2]*255 
                
                if depthBuffer.get(y * imageWidth + x):
                    # print("depthBuffer[y * imageWidth + x] = ",depthBuffer[y * imageWidth + x], "z = ",z)
                    if depthBuffer[y * imageWidth + x] > z:
                        
                        depthBuffer[y * imageWidth + x] = z
                        frameBuffer[y * imageWidth + x] = [int(r),int(g),int(b)]
                else:
                    depthBuffer[y * imageWidth + x] = z
                    frameBuffer[y * imageWidth + x] = [int(r),int(g),int(b)]
               

def cpp_vertexPlaneIntersectionPoint(insideVertex, outsideVertex, insidevertexW, outsideVertexW, intersectionPlane):
    x1 = insideVertex[0]
    y1 = insideVertex[1]
    z1 = insideVertex[2]
    w1 = insidevertexW

    x2 = outsideVertex[0]
    y2 = outsideVertex[1]
    z2 = outsideVertex[2]
    w2 = outsideVertexW
   
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
   
    t0 = tr_vertex_coordinates[edge1[0]][0]/ tr_vertex_ws[edge1[0]]
    t1 = tr_vertex_coordinates[edge1[0]][1]/ tr_vertex_ws[edge1[0]]
    t2 = tr_vertex_coordinates[edge1[0]][2]/ tr_vertex_ws[edge1[0]]
    
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
    
        t0 = tr_vertex_coordinates[secondVertex][0]/ tr_vertex_ws[secondVertex]
        t1 = tr_vertex_coordinates[secondVertex][1]/ tr_vertex_ws[secondVertex]
        t2 = tr_vertex_coordinates[secondVertex][2]/ tr_vertex_ws[secondVertex]

        if np.isnan(t0) or np.isnan(t1) or np.isnan(t2):
            # print("NAN value found")
            return
        
        raster1[0] = min(imageWidth - 1, (int)((t0 + 1) * 0.5 * imageWidth)) 
        raster1[1] = min(imageHeight - 1,(int)((1 - (t1 + 1) * 0.5) * imageHeight))
        raster1[2] = t2
   
        t0 = tr_vertex_coordinates[thirdVertex][0]/ tr_vertex_ws[thirdVertex]
        t1 = tr_vertex_coordinates[thirdVertex][1]/ tr_vertex_ws[thirdVertex]
        t2 = tr_vertex_coordinates[thirdVertex][2]/ tr_vertex_ws[thirdVertex]

        if np.isnan(t0) or np.isnan(t1) or np.isnan(t2):
            # print("NAN value found")
            return
        
        raster2[0] = min(imageWidth - 1, (int)((t0 + 1) * 0.5 * imageWidth)) 
        raster2[1] = min(imageHeight - 1,(int)((1 - (t1 + 1) * 0.5) * imageHeight))
        raster2[2] = t2
        
        currVertexColours = [currImageColours[firstVertex], currImageColours[secondVertex], currImageColours[thirdVertex]]
        
        drawTriangle2([raster0,raster1,raster2], currVertexColours)
        prviousFirstVertex =secondVertex
        secondVertex = thirdVertex
    
def renderATriangle(currTriangle,xp, yp,zp):
    vertex0 = nvertices[currTriangle*3+0]
    vertex1 = nvertices[currTriangle*3+1]
    vertex2 = nvertices[currTriangle*3+2]
    currTriangleVertices = [vertex0, vertex1,vertex2]
    
    v0Vertex = [vertices[currTriangleVertices[0]*3+0], vertices[currTriangleVertices[0]*3+1],vertices[currTriangleVertices[0]*3+2] ]
    v1Vertex = [vertices[currTriangleVertices[1]*3+0], vertices[currTriangleVertices[1]*3+1],vertices[currTriangleVertices[1]*3+2] ]
    v2Vertex = [vertices[currTriangleVertices[2]*3+0], vertices[currTriangleVertices[2]*3+1],vertices[currTriangleVertices[2]*3+2] ]


    if Counter(v0Vertex) == Counter(v1Vertex) or Counter(v0Vertex) == Counter(v2Vertex) or Counter(v1Vertex) == Counter(v2Vertex):
        return 0
   
      
    tempList1 = [tedges[(currTriangle) *6+0],tedges[(currTriangle) *6+1],tedges[(currTriangle) *6+2],
                 tedges[(currTriangle) *6+3],tedges[(currTriangle) *6+4],tedges[(currTriangle) *6+5]]
    for el in tempList1 :
        if el not in currTriangleVertices:           
            errorTriangleList.append(currTriangle)
            return 0
    
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
        
    currImageColours = dict()
    currImageColours.clear()
    
    currImageColours[0] = [vertColours[currTriangleVertices[0]*3+0],
                            vertColours[currTriangleVertices[0]*3+1],
                            vertColours[currTriangleVertices[0]*3+2]] 
    
    currImageColours[1] = [vertColours[currTriangleVertices[1]*3+0],
                            vertColours[currTriangleVertices[1]*3+1],
                            vertColours[currTriangleVertices[1]*3+2]] 
    
    currImageColours[2] = [vertColours[currTriangleVertices[2]*3+0],
                            vertColours[currTriangleVertices[2]*3+1],
                            vertColours[currTriangleVertices[2]*3+2]] 
    
    insideVertexDetailsToPPL = [] 
    numberOfFullyInsideVertices = 0
    numberOfIntersectingEdges = 0
    intersectingEdgeDataToPPL = []
    
    fullyInsideVerticesNumber = []
    intersectingVerticesNumber = []
    
    intersectingData = dict()
    intersectingData.clear()

    edges.clear()
    
    tr_num_of_vertices = 3
    tr_curr_num_of_vertex = 3
    tr_vertex_coordinates = np.zeros((100,3))
    tr_vertex_ws = [0]*100
    tr_vertex_outcodes = np.zeros((100,6))
    tr_vertices_set = []
    
    posXp = xp
    posYp = yp
    posZp = zp
    
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
    
    if(not any(outcodeP0)):
        currPixelValue = {}
        currPixelValue[0], temp = pixelValue(outValue0,outW0)        
        currPixelValue[1], temp = pixelValue(outValue1,outW1)        
        currPixelValue[2], temp = pixelValue(outValue2,outW2)
        
        drawTriangle2(currPixelValue, currImageColours)
    
    elif(bit0 or bit1 or bit2 or bit3 or bit4 or bit5):            
        pass
    else:
        num_of_planes = 6
        
        for currPlane in [5,0,1,2,3, ]:
            tr_outside_vertex_set = []
            num_of_outside_vertices = 0
            
            for currVert in tr_vertices_set:
                if(tr_vertex_outcodes[currVert][currPlane] == 1):
                    num_of_outside_vertices +=1
                    tr_outside_vertex_set.append(currVert)
            
            while(num_of_outside_vertices > 2):
                for currOutsideVert in tr_outside_vertex_set:               
                    edge1, edge2 = findEdges(currOutsideVert,edges)
                   
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
                edge1,edge2 = findEdges(outsideVertex, edges)

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
                    intersectionPoint1W = - intersectionPoint1W
                
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
        
        generateTriangles2(tr_vertex_coordinates, tr_vertex_ws, tr_num_of_vertices,edges,currTriangle, currImageColours)
        
        return

    
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

    for i in range(0,numOfTriangles):
        renderATriangle(i,xp,yp,zp)

    image = array.array('B', [1, 25, 24] * imageWidth * imageHeight)   
    maxval = 255
    for i in range(0, imageWidth * imageHeight):
        if frameBuffer.get(i):
            image[i * 3 + 0] = frameBuffer[i][0]
            image[i * 3 + 1] = frameBuffer[i][1]
            image[i * 3 + 2] = frameBuffer[i][2]
          
    ppm_header = f'P6 {imageWidth} {imageHeight} {maxval}\n' 

    # with open("images/py_" + str(currTriangle) +"_"+str(currImage)+ ".ppm", 'wb') as f:   
    with open(str(currImage)+".ppm", 'wb') as f:
        f.write(bytearray(ppm_header, 'ascii'))
        image.tofile(f)
    print("Image ", str(currImage)+".ppm saved successfully.")

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
