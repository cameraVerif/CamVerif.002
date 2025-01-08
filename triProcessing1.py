

import environment





vertices = environment.vertices
nvertices = environment.nvertices


numOfTriangles = environment.numOfTriangles

listToUpdate = [
                [ ], 
]

def writeEnvToAFile():

    outputFile = open("tempEnvOutput.txt", "w")

    for currTriangle in range(0, numOfTriangles):
        outputFile.write("\n###"+str(currTriangle)+"\n" )

        vertex0 = nvertices[currTriangle*3+0]
        vertex1 = nvertices[currTriangle*3+1]
        vertex2 = nvertices[currTriangle*3+2]
        currTriangleVertices = [vertex0, vertex1,vertex2]
        


        v0Vertex = [vertices[currTriangleVertices[0]*3+0], vertices[currTriangleVertices[0]*3+1],vertices[currTriangleVertices[0]*3+2] ]
        v1Vertex = [vertices[currTriangleVertices[1]*3+0], vertices[currTriangleVertices[1]*3+1],vertices[currTriangleVertices[1]*3+2] ]
        v2Vertex = [vertices[currTriangleVertices[2]*3+0], vertices[currTriangleVertices[2]*3+1],vertices[currTriangleVertices[2]*3+2] ]

        # print(v0Vertex)
        # print(v1Vertex)
        # print(v2Vertex)

        # outputFile.write(str(v0Vertex[0])+", "+str(v0Vertex[1])+", ", )

        
        outputFile.write(str(v0Vertex[0])+", "+str(v0Vertex[1])+", "+str(v0Vertex[2])+", \n")
        outputFile.write(str(v1Vertex[0])+", "+str(v1Vertex[1])+", "+str(v1Vertex[2])+", \n")
        outputFile.write(str(v2Vertex[0])+", "+str(v2Vertex[1])+", "+str(v2Vertex[2])+", \n")








