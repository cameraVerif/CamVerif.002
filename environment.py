

from z3 import *
from pyparma import *
import anytree
import scene
import camera

global dnnOutput, imagesMap, numOfVertices, vertices, numOfTriangles, nvertices, groupFrustum,\
        initFlag,imageGroup, outFileName, initialImageCount, imagePos, imageGroupStack,groupCount,groupFrustumFlag,\
        GloballoopCount, pplInputFileName, pplOutputFileName, initialZP, pplpathHullOutputFileName, \
        collisionCheckStartTriangle, pplSingleImageConstraintOutput,grpCubePoses,\
        x0,x1,y0,y1,z0,z1, intiFrusCons, initCubeCon, randomLoopLimit,numOfEdges,\
        canvasWidth, canvasHeight, focalLength,t,b,l,r,n,f,imageCons,groupRegionConsPPL,\
        groupCube,allInSameGrp,groupCubeZ3, targetRegionPolyhedron, \
        groupCubePostRegion,z3timeout,absStack, splitRegionPd, splitCount,spuriousCollisionCount,\
        imageWidth, imageHeight, depthOfTheInitialCube, A, numberOfSplit,refineCountNew,\
midPoints, processedMidPoints, spuriousCollisionData, vertColours, totalNumRefinment, networkName, netInterpreterPath
dnnOutput = dict()
imagesMap = dict()
imageGroup = dict()
imageCons = dict()
imagePos = dict()
groupRegionConsPPL = dict()
groupCube = dict()
groupCubeZ3 = dict()
groupCubePostRegion =dict()

midPoints = {}
processedMidPoints = {}
spuriousCollisionData = {}

splitRegionPd =dict()
splitCount = 0
numberOfSplit = 5
# numberOfRandomPointsToCheck = 10

#envs

totalNumRefinment=0

A = anytree.Node("A")

groupFrustum = {}

networkName = "OGmodel_pb_converted.onnx"
netInterpreterPath = "/home/habeeb/project2/alpha-beta-CROWN-mainfromLap/complete_verifier/abcrown.py"
refineCountNew = 0
spuriousCollisionCount=0



vertices = scene.vertices
numOfVertices = scene.numOfVertices
numOfTriangles = scene.numOfTriangles
nvertices = scene.nvertices
numOfEdges = scene.numOfEdges
vertColours = scene.vertColours
tedges = scene.tedges

z3timeout = 0

initFlag = 0
outFileName = "Env_11_12_8_abs_1_195_20Steps_1.txt"
initialImageCount = 0

allInSameGrp = dict()


groupCount =1
groupFrustumFlag = {}
GloballoopCount = 0
pplInputFileName = "imagesDataFromPython.txt"
pplOutputFileName = "constraintsFromPPL.txt"
pplpathHullOutputFileName = "pathHullOutput.txt"
collisionCheckStartTriangle = 4
pplSingleImageConstraintOutput = "singleImageconstraintsFromPPL.txt"

# imageWidth =49
# imageHeight = 49
# canvasWidth = 0.9872
# canvasHeight = 0.735
# focalLength = 35
# # t =0.35820895522388063
# # b =-0.35820895522388063
# # l =-0.35820895522388063
# # r =0.35820895522388063

imageWidth = camera.imageWidth
imageHeight = camera.imageHeight
canvasWidth = camera.filmApertureWidth
canvasHeight = camera.filmApertureHeight
focalLength = camera.focalLength

t = camera.t
b = camera.b
l = camera.l
r = camera.r


n = camera.nearClippingPlane
f = camera.farClippingPlane


grpCubePoses = dict()
imageGroupStack = []

absStack = []

xp0, yp0, zp0 = Reals('xp0 yp0 zp0')


def printLog(message):
    print(message)



intiFrusCons = [10*xp0>=1,100*xp0<=11,10*yp0>=45,100*yp0<=451, 10*zp0>=1945,100*zp0<=19451]
initCubeCon = And(10*xp0>=1,100*xp0<=11,10*yp0>=45,100*yp0<=451, 10*zp0>=1945,100*zp0<=19451)


xp0 = Variable(0)
yp0 = Variable(1)
zp0 = Variable(2)
pd3 = NNC_Polyhedron(3)
pd3.add_constraint(10*xp0>=1)
pd3.add_constraint(100*xp0<=11)
pd3.add_constraint(10*yp0>=45)
pd3.add_constraint(100*yp0<=451)
pd3.add_constraint(10*zp0>=1945)
pd3.add_constraint(100*zp0<=19451)

midPoints["A"] = [0.1,4.5,194.5] ##front left bottom corner of the  initial region, (anyone point from the region is enough)
currentMidPoint = midPoints["A"]
currentMidPointString = str(currentMidPoint[0])+"_"+str(currentMidPoint[1])+"_"+str(currentMidPoint[2])
processedMidPoints[currentMidPointString] = "A"



depthOfTheInitialCube = .01 ##in meters







groupCube["G_0"] = pd3.minimized_constraints()
groupCube["G_1"] = pd3.minimized_constraints()
groupCube["G_2"] = pd3.minimized_constraints()

groupCube["A"] = pd3.minimized_constraints()
# groupCube["A_1"] = pd3.minimized_constraints()
# groupCube["A_2"] = pd3.minimized_constraints()



groupCubeZ3["G_0"] = initCubeCon
groupCubeZ3["G_1"] = initCubeCon
groupCubeZ3["G_2"] = initCubeCon

groupCubeZ3["A"] = initCubeCon


groupCube["initCubeCon"] = pd3.minimized_constraints()
groupCubePostRegion["initCubeCon"] = pd3.minimized_constraints()
groupCubePostRegion["A"] = pd3.minimized_constraints()




targetRegionPolyhedron = NNC_Polyhedron(3)
targetRegionPolyhedron.add_constraint(xp0>=-200)
targetRegionPolyhedron.add_constraint(xp0<=200)
targetRegionPolyhedron.add_constraint(yp0>=4)
targetRegionPolyhedron.add_constraint(yp0<=5)
targetRegionPolyhedron.add_constraint(zp0>=100)
targetRegionPolyhedron.add_constraint(zp0<=169)





z0 = 194.5
z1 = 194.51

randomLoopLimit = 1000
