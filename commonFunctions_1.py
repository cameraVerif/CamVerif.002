

import camera
import scene
import environment




n =camera.nearClippingPlane
f =camera.farClippingPlane
t =camera.t
b =camera.b
l =camera.l
r =camera.r


mProj = [
        [2 * n / (r - l), 0, 0, 0],
        [0,2 * n / (t - b),0,0],
        [(r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n), -1 ],
        [0,0,-2 * f * n / (f - n),0]
    ]


def getSceneData():
    return scene.numOfTriangles, scene.numOfVertices, scene.numOfEdges

def getVertices():
    return scene.vertices

def getNVertices():
    return scene.nvertices

def getVertColours():
    return scene.vertColours

def getTEdges():
    return scene.tedges

def getTriangleVertices(triangleIndex):
    vertices = getVertices()
    nvertices = getNVertices()
    vertColours = getVertColours()
   
    triangleVertices = []
    
    for i in range(3):
        currVertexIndex = nvertices[triangleIndex*3+i]
        currVertex = vertices[currVertexIndex*3:currVertexIndex*3+3]
        currVertexColour = vertColours[currVertexIndex*3:currVertexIndex*3+3]
        triangleVertices.append([currVertex, currVertexColour])
    
    return triangleVertices




