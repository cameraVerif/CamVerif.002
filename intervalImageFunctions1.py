
import environment
import math
import array

imageWidth = environment.imageWidth
imageHeight = environment.imageHeight
frameBuffer = dict()
nvertices = environment.nvertices
vertColours = environment.vertColours


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


def edgeFunction(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])     
    





def drawTriangle(raster0,raster1,raster2,currTriangle,d0,d1,d2,d3,d4,d5, currTriangleIntervalImage, currVertexColours, currImg):
    
    
    
    
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
        v0MinDepth = d0
        v0MaxDepth = d1
    elif(minAngle == angle1):
        tempV0 = raster1
        v0flag =1
        v0MinDepth = d2
        v0MaxDepth = d3
    else:
        tempV0 = raster2
        v0flag =2
        v0MinDepth = d4
        v0MaxDepth = d5
    
 
    if(v0flag == 0):
        if(angle1<=angle2):
            tempV1 = raster1
            v1flag =1
            v1MinDepth = d2
            v1MaxDepth = d3
        else:
            tempV1 = raster2
            v1flag =2
            v1MinDepth = d4
            v1MaxDepth = d5
    elif(v0flag == 1):
        if(angle0<=angle2):
            tempV1 = raster0
            v1flag =0
            v1MinDepth = d0
            v1MaxDepth = d1
        else:
            tempV1 = raster2
            v1flag =2
            v1MinDepth = d4
            v1MaxDepth = d5
    else:
        if(angle0<=angle1):
            tempV1 = raster0
            v1flag =0
            v1MinDepth = d0
            v1MaxDepth = d1
        else:
            tempV1 = raster1
            v1flag =1
            v1MinDepth = d2
            v1MaxDepth = d3
    
    
    if(v0flag != 0 and v1flag != 0 ):
        tempV2 = raster0
        v2MinDepth = d0
        v2MaxDepth = d1
    elif(v0flag != 1 and v1flag != 1 ):
        tempV2 = raster1
        v2MinDepth = d2
        v2MaxDepth = d3
    else:
        tempV2 = raster2
        v2MinDepth = d4
        v2MaxDepth = d5
    
    
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
                storeZasDepth = z
                
                
                
                r = w0 * currVertexColours[0][0]*255 + w1 * currVertexColours[1][0]*255 + w2 * currVertexColours[2][0]*255 
                g = w0 * currVertexColours[0][1]*255 + w1 * currVertexColours[1][1]*255 + w2 * currVertexColours[2][1]*255
                b = w0 * currVertexColours[0][2]*255 + w1 * currVertexColours[1][2]*255 + w2 * currVertexColours[2][2]*255
                
                currMinDepth = v0MinDepth * w0 + v1MinDepth * w1 + v2MinDepth * w2
                currMaxDepth = v0MaxDepth * w0 + v1MaxDepth * w1 + v2MaxDepth * w2
               
                frameBuffer[y * imageWidth + x] = [int(r),int(g),int(b)]
                
                if currTriangleIntervalImage.get(y*imageWidth+x):
                    # print(y*imageWidth+x, "already exists")
                    currValues = currTriangleIntervalImage[y*imageWidth+x]
                    currValues.append([r,g,b,currMinDepth,currMaxDepth])
                    currTriangleIntervalImage[y*imageWidth+x] = currValues
                else:
                    currTriangleIntervalImage[y*imageWidth+x] = [[r,g,b,currMinDepth,currMaxDepth]]

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
    
    
    

   






def computeSingleImage(currImageName, numberOfInvRegions, dataToComputeDepth, depthInformation, 
            tr_vertices_set, tr_vertex_coordinates, tr_vertex_ws, edges, numberOfFullyInsideVertices, numberOfIntersectingEdges,tr_num_of_vertices, 
            currTriangle, currTriangleIntervalImage, currImageColours):
    
    # print("Computing image: " + currImageName)
    # print("Region Number: " + str(numberOfInvRegions))
    # print("Data to compute depth: " + str(dataToComputeDepth))
    # print("Depth information: " + str(depthInformation))
    # print("Number of fully inside vertices: " + str(numberOfFullyInsideVertices))
    # print("Number of intersecting edges: " + str(numberOfIntersectingEdges))
    # print("tr_vertices_set: " + str(tr_vertices_set))
    # # print("tr_vertex_coordinates: " + str(tr_vertex_coordinates))
    # print("tr_vertex_ws: " + str(tr_vertex_ws))
    # print("edges: " + str(edges))
    
    frameBuffer.clear()
    
    raster0 = [0,0,0]
    raster1 = [0,0,0]
    raster2 = [0,0,0]
    
    edge1 = edges[0]
   
    t0 = tr_vertex_coordinates[edge1[0]][0]/ tr_vertex_ws[edge1[0]]
    t1 = tr_vertex_coordinates[edge1[0]][1]/ tr_vertex_ws[edge1[0]]
    t2 = tr_vertex_coordinates[edge1[0]][2]/ tr_vertex_ws[edge1[0]]
    
    d0 = depthInformation[edge1[0]][1]
    d1 = depthInformation[edge1[0]][2]
    
    
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
        
        raster1[0] = min(imageWidth - 1, (int)((t0 + 1) * 0.5 * imageWidth)) 
        raster1[1] = min(imageHeight - 1,(int)((1 - (t1 + 1) * 0.5) * imageHeight))
        raster1[2] = t2
        
        
        d2 = depthInformation[secondVertex][1]
        d3 = depthInformation[secondVertex][2]
        
        t0 = tr_vertex_coordinates[thirdVertex][0]/ tr_vertex_ws[thirdVertex]
        t1 = tr_vertex_coordinates[thirdVertex][1]/ tr_vertex_ws[thirdVertex]
        t2 = tr_vertex_coordinates[thirdVertex][2]/ tr_vertex_ws[thirdVertex]
        
        raster2[0] = min(imageWidth - 1, (int)((t0 + 1) * 0.5 * imageWidth)) 
        raster2[1] = min(imageHeight - 1,(int)((1 - (t1 + 1) * 0.5) * imageHeight))
        raster2[2] = t2
        
       
        d4 = depthInformation[thirdVertex][1]
        d5 = depthInformation[thirdVertex][2]

        # print("Draw traingle : ", raster0,raster1,raster2)
        # 
        
        currVertexColours = [currImageColours[firstVertex], currImageColours[secondVertex], currImageColours[thirdVertex]]
        # print("Vertex Colours: " + str(currVertexColours))
        
        drawTriangle(raster0,raster1,raster2,currTriangle,d0,d1,d2,d3,d4,d5, currTriangleIntervalImage, currVertexColours, currImg)
        # print("Drawing Triangle Done: " + str(currTriangle))

        prviousFirstVertex =secondVertex
        secondVertex = thirdVertex
    
    
    # image = array.array('B', [1, 25, 24] * imageWidth * imageHeight)   
    # maxval = 255
    # for i in range(0, imageWidth * imageHeight):
    #     if frameBuffer.get(i):
    #         image[i * 3 + 0] = frameBuffer[i][0]
    #         image[i * 3 + 1] = frameBuffer[i][1]
    #         image[i * 3 + 2] = frameBuffer[i][2]
    # ppm_header = f'P6 {imageWidth} {imageHeight} {maxval}\n' 
    
    # with open("images/py_" + str(currTriangle) +"_"+str(currImg)+ ".ppm", 'wb') as f:   
    #     f.write(bytearray(ppm_header, 'ascii'))
    #     image.tofile(f)

 
 

def updateSingleIntervalImage(currTriangleIntervalImage,  currTriangle, minmaxDepths, centerPointImage):

    vertex0 = nvertices[currTriangle*3+0]
    # vertex1 = nvertices[currTriangle*3+1]
    # vertex2 = nvertices[currTriangle*3+2]
    # currTriangleVertices = [vertex0, vertex1,vertex2]

    r = vertColours[vertex0*3+0]*255
    g = vertColours[vertex0*3+1]*255
    b = vertColours[vertex0*3+2]*255
    for currPixel in centerPointImage:
        if currTriangleIntervalImage.get(currPixel):
            # print(y*imageWidth+x, "already exists")
            currValues = currTriangleIntervalImage[currPixel]
            currValues.append([r,g,b,minmaxDepths[0],minmaxDepths[1]])
            currTriangleIntervalImage[currPixel] = currValues
        else:
            currTriangleIntervalImage[currPixel] = [[r,g,b,minmaxDepths[0],minmaxDepths[1]]]


    
    
    
    