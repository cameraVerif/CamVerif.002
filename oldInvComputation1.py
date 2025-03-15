from z3 import *

from time import sleep
import environment
import pythonRenderAnImageDict_1


currWidth = 0.01


vertices = environment.vertices
nvertices = environment.nvertices




xp0, yp0, zp0 = Reals('xp0 yp0 zp0')





def checkAllImages(centerPointImage, currWidth, posXp, posYp, posZp, currTriangle ):


    # centerPixelLength = len(centerPointImage)
    currScore = 0
    pbotLeftFront = [posXp - currWidth, posYp - currWidth, posZp - currWidth]
    pbotLeftFrontImage = pythonRenderAnImageDict_1.renderATrianglePixels( pbotLeftFront[0], pbotLeftFront[1], pbotLeftFront[2], currTriangle)

    if(centerPointImage == pbotLeftFrontImage):
        # print("image equals with pbotLeftFrontImage")
        # print(len(pbotLeftFrontImage))
        currScore += 1

    else:
        # print("inot pbotLeftFrontImage")
        return currScore
    

    pbotLeftBack = [posXp - currWidth, posYp - currWidth, posZp + currWidth]
    pbotLeftBackImage = pythonRenderAnImageDict_1.renderATrianglePixels( pbotLeftBack[0], pbotLeftBack[1], pbotLeftBack[2], currTriangle)

    if(centerPointImage == pbotLeftBackImage):
        # print("image equals with pbotLeftBackImage")
        # print(len(pbotLeftBackImage))
        currScore += 1

    else:
        # print("inot pbotRightFront")
        return currScore
    

    
    pbotRightFront = [posXp + currWidth, posYp - currWidth, posZp - currWidth]
    pbotRightFrontImage = pythonRenderAnImageDict_1.renderATrianglePixels( pbotRightFront[0], pbotRightFront[1], pbotRightFront[2], currTriangle)

    if(centerPointImage == pbotRightFrontImage):
        # print("image equals with pbotRightFront")
        # print(len(pbotRightFrontImage))
        currScore += 1
    else:
        # print("inot pbotRightFront")
        return currScore
    

    
    pbotRightBack = [posXp + currWidth, posYp - currWidth, posZp + currWidth]
    pbotRightBackImage = pythonRenderAnImageDict_1.renderATrianglePixels( pbotRightBack[0], pbotRightBack[1], pbotRightBack[2], currTriangle)

    if(centerPointImage == pbotRightBackImage):
        # print("image equals with pbotRightBackImage")
        # print(len(pbotRightBackImage))
        currScore += 1
    else:
        # print("inot pbotRightBackImage")
        return currScore
    

    pTopLeftFront = [posXp - currWidth, posYp + currWidth, posZp - currWidth]
    pTopLeftFrontImage = pythonRenderAnImageDict_1.renderATrianglePixels( pTopLeftFront[0], pTopLeftFront[1], pTopLeftFront[2], currTriangle)

    if(centerPointImage == pTopLeftFrontImage):
        # print("image equals with pTopLeftFrontImage")
        # print(len(pTopLeftFrontImage))
        currScore += 1
    else:
        # print("inot pTopLeftFrontImage")
        return currScore
    
    

    pTopLeftBack = [posXp - currWidth, posYp + currWidth, posZp + currWidth]
    pTopLeftBackImage = pythonRenderAnImageDict_1.renderATrianglePixels( pTopLeftBack[0], pTopLeftBack[1], pTopLeftBack[2], currTriangle)

    if(centerPointImage == pTopLeftBackImage):
        # print("image equals with pTopLeftBackImage")
        # print(len(pTopLeftBackImage))
        currScore += 1
    else:
        # print("inot pTopLeftBackImage")
        return currScore
    
    
    pTopRightFront = [posXp + currWidth, posYp + currWidth, posZp - currWidth]
    pTopRightFrontImage = pythonRenderAnImageDict_1.renderATrianglePixels( pTopRightFront[0], pTopRightFront[1], pTopRightFront[2], currTriangle)

    if(centerPointImage == pTopRightFrontImage):
        # print("image equals with pTopRightFront")
        # print(len(pTopRightFrontImage))
        currScore += 1
    else:
        # print("inot pTopRightFront")
        return currScore
    
    
    pTopRightBack = [posXp + currWidth, posYp + currWidth, posZp + currWidth]
    pTopRightBackImage = pythonRenderAnImageDict_1.renderATrianglePixels( pTopRightBack[0], pTopRightBack[1], pTopRightBack[2], currTriangle)

    if(centerPointImage == pTopRightBackImage):
        # print("image equals with pTopLeftFrontImage")
        # print(len(pTopRightBackImage))
        currScore += 1
    else:
        # print("inot pTopLeftFrontImage")
        return currScore
    
    return currScore
    










def computeInv( posXp, posYp, posZp, currTriangle,m):

    currWidth = 0.01
    foundR = 0
    
    # print("oldcomputeInva ", posXp, posYp, posZp, currTriangle,m)
    centerPointImage = pythonRenderAnImageDict_1.renderATrianglePixels( posXp, posYp, posZp, currTriangle)
    # print(len(centerPointImage))

    returnScore = 0
    maxDivCount = 100
    currDivCount = 0
    successFlag = 0

    while(currDivCount < maxDivCount):
        returnScore = checkAllImages(centerPointImage, currWidth, posXp, posYp, posZp, currTriangle )
        # print("returnScore = ", returnScore)
        
        if returnScore == 8:
            # print(currWidth, posXp, posYp, posZp, currTriangle )
            successFlag = 1
            break
        else:
            currWidth /=2
            currDivCount +=1
            # print("Not maching image found, refining again")
            # print(" currDivCount = ",currDivCount, "currWidth = ",currWidth )

            # print("currTriangle = ", currTriangle)


    if successFlag == 0:
        # print("Failed all tries")
        return 0, "False", [0,0] , []
    else:
        # print("Found Region , currWidth = ",currWidth, " currDivCount = ",currDivCount )
        consOfReg =   str(posXp-currWidth) +"<=xp0, xp0<="+ str(posXp+currWidth)+"," \
                    + str(posYp-currWidth) +"<=yp0, yp0<="+ str(posYp+currWidth)+"," \
                    + str(posZp-currWidth) +"<=zp0, zp0<="+ str(posZp+currWidth)
        # consOfReg =   str(m[xp0]-currWidth) +"<=xp0, xp0<="+ str(m[xp0]+currWidth)+"," \
        #             + str(m[yp0]-currWidth) +"<=yp0, yp0<="+ str(m[yp0]+currWidth)+"," \
        #             + str(m[zp0]-currWidth) +"<=zp0, zp0<="+ str(m[zp0]+currWidth)
        # print(consOfReg)

        vertex0 = nvertices[currTriangle*3+0]
        vertex1 = nvertices[currTriangle*3+1]
        vertex2 = nvertices[currTriangle*3+2]
        currTriangleVertices = [vertex0, vertex1,vertex2]
        v0Vertex = [vertices[currTriangleVertices[0]*3+0], vertices[currTriangleVertices[0]*3+1],vertices[currTriangleVertices[0]*3+2] ]
        v1Vertex = [vertices[currTriangleVertices[1]*3+0], vertices[currTriangleVertices[1]*3+1],vertices[currTriangleVertices[1]*3+2] ]
        v2Vertex = [vertices[currTriangleVertices[2]*3+0], vertices[currTriangleVertices[2]*3+1],vertices[currTriangleVertices[2]*3+2] ]


        minDepth = min(abs(posZp - v0Vertex[2]), abs(posZp - v1Vertex[2]), abs(posZp - v2Vertex[2]))
        maxDepth = max(abs(posZp - v0Vertex[2]), abs(posZp - v1Vertex[2]), abs(posZp - v2Vertex[2]))+currWidth



        return 1, consOfReg, [minDepth, maxDepth], centerPointImage




    


    
    
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
    


# def invUsingConstraints(posXp, posYp, posZp, currTriangle,m):  

















