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
        currScore += 1

    else:
        return currScore
    

    pbotLeftBack = [posXp - currWidth, posYp - currWidth, posZp + currWidth]
    pbotLeftBackImage = pythonRenderAnImageDict_1.renderATrianglePixels( pbotLeftBack[0], pbotLeftBack[1], pbotLeftBack[2], currTriangle)

    if(centerPointImage == pbotLeftBackImage):
        currScore += 1

    else:
        return currScore
    

    
    pbotRightFront = [posXp + currWidth, posYp - currWidth, posZp - currWidth]
    pbotRightFrontImage = pythonRenderAnImageDict_1.renderATrianglePixels( pbotRightFront[0], pbotRightFront[1], pbotRightFront[2], currTriangle)

    if(centerPointImage == pbotRightFrontImage):
        currScore += 1
    else:
        return currScore
    

    
    pbotRightBack = [posXp + currWidth, posYp - currWidth, posZp + currWidth]
    pbotRightBackImage = pythonRenderAnImageDict_1.renderATrianglePixels( pbotRightBack[0], pbotRightBack[1], pbotRightBack[2], currTriangle)

    if(centerPointImage == pbotRightBackImage):
        currScore += 1
    else:
        return currScore
    

    pTopLeftFront = [posXp - currWidth, posYp + currWidth, posZp - currWidth]
    pTopLeftFrontImage = pythonRenderAnImageDict_1.renderATrianglePixels( pTopLeftFront[0], pTopLeftFront[1], pTopLeftFront[2], currTriangle)

    if(centerPointImage == pTopLeftFrontImage):
        currScore += 1
    else:
        return currScore
    
    

    pTopLeftBack = [posXp - currWidth, posYp + currWidth, posZp + currWidth]
    pTopLeftBackImage = pythonRenderAnImageDict_1.renderATrianglePixels( pTopLeftBack[0], pTopLeftBack[1], pTopLeftBack[2], currTriangle)

    if(centerPointImage == pTopLeftBackImage):
        currScore += 1
    else:
        return currScore
    
    
    pTopRightFront = [posXp + currWidth, posYp + currWidth, posZp - currWidth]
    pTopRightFrontImage = pythonRenderAnImageDict_1.renderATrianglePixels( pTopRightFront[0], pTopRightFront[1], pTopRightFront[2], currTriangle)

    if(centerPointImage == pTopRightFrontImage):
        currScore += 1
    else:
        return currScore
    
    
    pTopRightBack = [posXp + currWidth, posYp + currWidth, posZp + currWidth]
    pTopRightBackImage = pythonRenderAnImageDict_1.renderATrianglePixels( pTopRightBack[0], pTopRightBack[1], pTopRightBack[2], currTriangle)

    if(centerPointImage == pTopRightBackImage):
        currScore += 1
    else:
        return currScore
    
    return currScore
    










def computeInv( posXp, posYp, posZp, currTriangle,m):

    currWidth = 0.01
    foundR = 0
    

    centerPointImage = pythonRenderAnImageDict_1.renderATrianglePixels( posXp, posYp, posZp, currTriangle)
   

    returnScore = 0
    maxDivCount = 100
    currDivCount = 0
    successFlag = 0

    while(currDivCount < maxDivCount):
        returnScore = checkAllImages(centerPointImage, currWidth, posXp, posYp, posZp, currTriangle )
        
        
        if returnScore == 8:
            successFlag = 1
            break
        else:
            currWidth /=2
            currDivCount +=1
            


    if successFlag == 0:
        return 0, "False", [0,0] , []
    else:
        consOfReg =   str(posXp-currWidth) +"<=xp0, xp0<="+ str(posXp+currWidth)+"," \
                    + str(posYp-currWidth) +"<=yp0, yp0<="+ str(posYp+currWidth)+"," \
                    + str(posZp-currWidth) +"<=zp0, zp0<="+ str(posZp+currWidth)
        # consOfReg =   str(m[xp0]-currWidth) +"<=xp0, xp0<="+ str(m[xp0]+currWidth)+"," \
        #             + str(m[yp0]-currWidth) +"<=yp0, yp0<="+ str(m[yp0]+currWidth)+"," \
        #             + str(m[zp0]-currWidth) +"<=zp0, zp0<="+ str(m[zp0]+currWidth)

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
    
    











