

from z3 import *
from pyparma import *
import anytree

import camera
import scene

global dnnOutput, imagesMap, numOfVertices, vertices, numOfTriangles, nvertices, groupFrustum,\
        initFlag,imageGroup, outFileName, initialImageCount, imagePos, imageGroupStack,groupCount,groupFrustumFlag,\
                GloballoopCount, pplInputFileName, pplOutputFileName, initialZP, pplpathHullOutputFileName, \
                        collisionCheckStartTriangle, pplSingleImageConstraintOutput,grpCubePoses,\
                                x0,x1,y0,y1,z0,z1, intiFrusCons, initCubeCon, randomLoopLimit,numOfEdges,\
                                        canvasWidth, canvasHeight, focalLength,t,b,l,r,n,f,imageCons,groupRegionConsPPL,\
                                                groupCube,allInSameGrp,groupCubeZ3, targetRegionPolyhedron, \
                                                        groupCubePostRegion,z3timeout,absStack, splitRegionPd, splitCount,\
                                                                imageWidth, imageHeight, depthOfTheInitialCube, A, numberOfSplit,\
                                                                        midPoints, processedMidPoints, spuriousCollisionData, vertColours, nnenumFlag, refineCount,\
                 initRegionMinMaxValues,initRegionCornerPoints, regionMinMaxValues, regionCornerPoints, totalNumRefinment


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
numberOfSplit = 2
# numberOfRandomPointsToCheck = 10
refineCount =0
#envs
nnenumFlag =0

A = anytree.Node("A")

vertices = scene.vertices
numOfVertices = scene.numOfVertices
numOfTriangles = scene.numOfTriangles
nvertices = scene.nvertices
numOfEdges = scene.numOfEdges
vertColours = scene.vertColours
tedges = scene.tedges


regionMinMaxValues = {}  
regionCornerPoints = {}

totalNumRefinment =0


groupFrustum = {}
# numOfTriangles = 4*4+250# 290+4*4 # 290 # 4+1+1
# numOfVertices = 4*6+(250*3)#+6*4# 290*3 # 6+3+3
# numOfEdges =  4*9+(250*3)#+9*4 #290*3# 9+3+3

def printLog(message):
    print(message)


# #buildings2
# numOfTriangles = 1729# 290+4*4 # 290 # 4+1+1
# numOfVertices = 2765#+6*4# 290*3 # 6+3+3
# numOfEdges =  5187#+9*4 #290*3# 9+3+3


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

# imageWidth = camera.imageWidth
# imageHeight = camera.imageHeight
# canvasWidth = camera.filmApertureWidth
# canvasHeight = camera.filmApertureHeight
# focalLength = camera.focalLength
# t =0.35820895522388063
# b =-0.35820895522388063
# l =-0.35820895522388063
# r =0.35820895522388063
# t =0.358213
# b =-0.358213
# l =-0.358213
# r =0.358213
# t =0.358
# b =-0.358
# l =-0.358
# r =0.358
t = camera.t
b = camera.b
l = camera.l
r = camera.r
imageWidth = camera.imageWidth
imageHeight = camera.imageHeight

n = camera.nearClippingPlane
f = camera.farClippingPlane

canvasWidth = camera.filmApertureWidth
canvasHeight = camera.filmApertureHeight
focalLength = camera.focalLength


grpCubePoses = dict()
imageGroupStack = []

absStack = []

xp0, yp0, zp0 = Reals('xp0 yp0 zp0')






###########################ORIGINAL#################################

# intiFrusCons = [10*xp0>=1,100*xp0<=11,10*yp0>=45,100*yp0<=451, 10*zp0>=1945,100*zp0<=19451]
# initCubeCon = And(10*xp0>=1,100*xp0<=11,10*yp0>=45,100*yp0<=451, 10*zp0>=1945,100*zp0<=19451)


# xp0 = Variable(0)
# yp0 = Variable(1)
# zp0 = Variable(2)
# pd3 = NNC_Polyhedron(3)
# pd3.add_constraint(10*xp0>=1)
# pd3.add_constraint(100*xp0<=11)
# pd3.add_constraint(10*yp0>=45)
# pd3.add_constraint(100*yp0<=451)
# pd3.add_constraint(10*zp0>=1945)
# pd3.add_constraint(100*zp0<=19451)

# midPoints["A"] = [0.1,4.5,194.5]
# currentMidPoint = midPoints["A"]
# currentMidPointString = str(currentMidPoint[0])+"_"+str(currentMidPoint[1])+"_"+str(currentMidPoint[2])
# processedMidPoints[currentMidPointString] = "A"



# initRegionMinMaxValues = [.1,.11,4.5,4.51,194.5,194.51]
# initRegionCornerPoints = [[0.1,4.5,194.5], [0.11,4.5,194.5], [0.1,4.51,194.5], [0.11,4.51,194.5],
#                           [0.1,4.5,194.51], [0.11,4.5,194.51], [0.1,4.51,194.51], [0.11,4.51,194.51]
#                           ]

# regionCornerPoints["A"] = initRegionCornerPoints
# regionMinMaxValues["A"] = initRegionMinMaxValues

# depthOfTheInitialCube = .01

#########################ORIGINAL_END#################################



#######################PineTree Start################################


#intiFrusCons = [10*xp0>=36,10000*xp0<=36025,10*yp0>=45,10000*yp0<=45025, 1000*zp0>=169438,10000*zp0<=1694404]
#initCubeCon = And(10*xp0>=36,10000*xp0<=36025,10*yp0>=45,10000*yp0<=45025, 1000*zp0>=169438,10000*zp0<=1694404)


#xp0 = Variable(0)
#yp0 = Variable(1)
#zp0 = Variable(2)
#pd3 = NNC_Polyhedron(3)
#pd3.add_constraint(10*xp0>=36)
#pd3.add_constraint(10000*xp0<=36025)
#pd3.add_constraint(10*yp0>=45)
#pd3.add_constraint(10000*yp0<=45025)
#pd3.add_constraint(1000*zp0>=169438)
#pd3.add_constraint(10000*zp0<=1694404)

#midPoints["A"] = [3.6,4.5,169.348]
#currentMidPoint = midPoints["A"]
#currentMidPointString = str(currentMidPoint[0])+"_"+str(currentMidPoint[1])+"_"+str(currentMidPoint[2])
#processedMidPoints[currentMidPointString] = "A"





# intiFrusCons = [10*xp0>=1,100*xp0<=11,10*yp0>=45,100*yp0<=451, 10*zp0>=1755,100*zp0<=17551]
# initCubeCon = And(10*xp0>=1,100*xp0<=11,10*yp0>=45,100*yp0<=451, 10*zp0>=1755,100*zp0<=17551)


# xp0 = Variable(0)
# yp0 = Variable(1)
# zp0 = Variable(2)
# pd3 = NNC_Polyhedron(3)
# pd3.add_constraint(10*xp0>=1)
# pd3.add_constraint(100*xp0<=11)
# pd3.add_constraint(10*yp0>=45)
# pd3.add_constraint(100*yp0<=451)
# pd3.add_constraint(10*zp0>=1755)
# pd3.add_constraint(100*zp0<=17551)

# midPoints["A"] = [0.1,4.5,175.5]
# currentMidPoint = midPoints["A"]
# currentMidPointString = str(currentMidPoint[0])+"_"+str(currentMidPoint[1])+"_"+str(currentMidPoint[2])
# processedMidPoints[currentMidPointString] = "A"


# initRegionMinMaxValues = [.1,.11,4.5,4.51,175.5,175.51]
# initRegionCornerPoints = [[0.1,4.5,175.5], [0.11,4.5,175.5], [0.1,4.51,175.5], [0.11,4.51,175.5],
#                            [0.1,4.5,175.51], [0.11,4.5,175.51], [0.1,4.51,175.51], [0.11,4.51,175.51]
#                            ]

# regionCornerPoints["A"] = initRegionCornerPoints
# regionMinMaxValues["A"] = initRegionMinMaxValues

# depthOfTheInitialCube = .01

#######################PineTree End################################

###############BUilding start  here##################
intiFrusCons = [10*xp0>=1,100*xp0<=11,10*yp0>=45,100*yp0<=451, 10*zp0>=1645,100*zp0<=16451]
initCubeCon = And(10*xp0>=1,100*xp0<=11,10*yp0>=45,100*yp0<=451, 10*zp0>=1645,100*zp0<=16451)


xp0 = Variable(0)
yp0 = Variable(1)
zp0 = Variable(2)
pd3 = NNC_Polyhedron(3)
pd3.add_constraint(10*xp0>=1)
pd3.add_constraint(100*xp0<=11)
pd3.add_constraint(10*yp0>=45)
pd3.add_constraint(100*yp0<=451)
pd3.add_constraint(10*zp0>=1645)
pd3.add_constraint(100*zp0<=16451)

midPoints["A"] = [0.1,4.5,164.5]
currentMidPoint = midPoints["A"]
currentMidPointString = str(currentMidPoint[0])+"_"+str(currentMidPoint[1])+"_"+str(currentMidPoint[2])
processedMidPoints[currentMidPointString] = "A"

initRegionMinMaxValues = [.1,.11,4.5,4.51,164.5,164.51]
initRegionCornerPoints = [[0.1,4.5,164.5], [0.11,4.5,164.5], [0.1,4.51,164.5], [0.11,4.51,164.5],
                           [0.1,4.5,164.51], [0.11,4.5,164.51], [0.1,4.51,164.51], [0.11,4.51,164.51]
                           ]

regionCornerPoints["A"] = initRegionCornerPoints
regionMinMaxValues["A"] = initRegionMinMaxValues
depthOfTheInitialCube = .01

#####################collision testing####### Pinetree##########
# intiFrusCons = [10*xp0>=-94,100*xp0<=-939,10*yp0>=45,100*yp0<=451, 1000*zp0>=141582,1000*zp0<=141592]
# initCubeCon = And(10*xp0>=-94,100*xp0<=-939,10*yp0>=45,100*yp0<=451, 1000*zp0>=141582,1000*zp0<=141592)


# xp0 = Variable(0)
# yp0 = Variable(1)
# zp0 = Variable(2)
# pd3 = NNC_Polyhedron(3)
# pd3.add_constraint(10*xp0>=-94)
# pd3.add_constraint(100*xp0<=-939)
# pd3.add_constraint(10*yp0>=45)
# pd3.add_constraint(100*yp0<=451)
# pd3.add_constraint(1000*zp0>=141582)
# pd3.add_constraint(1000*zp0<=141592)

# midPoints["A"] = [-9.4,4.5,141.582]


# initRegionMinMaxValues = [-9.4,-9.39,4.5,4.51,141.582,141.592]
# initRegionCornerPoints = [[-9.4,4.5,141.582], [-9.39,4.5,141.582], [-9.4,4.51,141.582], [-9.39,4.51,141.582],
#                            [-9.4,4.5,141.592], [-9.39,4.5,141.592], [-9.4,4.51,141.592], [-9.39,4.51,141.592]
#                            ]


# regionCornerPoints["A"] = initRegionCornerPoints
# regionMinMaxValues["A"] = initRegionMinMaxValues

# depthOfTheInitialCube = .01
#####################collision testing####### Pinetree# end######### 


#somany safe paths pine tree
# intiFrusCons = [10*xp0>=1,100*xp0<=11,10*yp0>=45,100*yp0<=451, 10*zp0>=1545,100*zp0<=15451]
# initCubeCon = And(10*xp0>=1,100*xp0<=11,10*yp0>=45,100*yp0<=451, 10*zp0>=1545,100*zp0<=15451)


# xp0 = Variable(0)
# yp0 = Variable(1)
# zp0 = Variable(2)
# pd3 = NNC_Polyhedron(3)
# pd3.add_constraint(10*xp0>=1)
# pd3.add_constraint(100*xp0<=11)
# pd3.add_constraint(10*yp0>=45)
# pd3.add_constraint(100*yp0<=451)
# pd3.add_constraint(10*zp0>=1545)
# pd3.add_constraint(100*zp0<=15451)

# midPoints["A"] = [0.1,4.5,154.5]

# currentMidPoint = midPoints["A"]
# currentMidPointString = str(currentMidPoint[0])+"_"+str(currentMidPoint[1])+"_"+str(currentMidPoint[2])
# processedMidPoints[currentMidPointString] = "A"

# # initRegionMinMaxValues = [.1,.11,4.5,4.51,154.5,154.51]
# # initRegionCornerPoints = [[0.1,4.5,154.5], [0.11,4.5,154.5], [0.1,4.51,154.5], [0.11,4.51,154.5],
# #                            [0.1,4.5,154.51], [0.11,4.5,154.51], [0.1,4.51,154.51], [0.11,4.51,154.51]
# #                            ]
# depthOfTheInitialCube = .01
#########################################################




# intiFrusCons = [10*xp0>=1,100*xp0<=11,10*yp0>=45,100*yp0<=451, 10*zp0>=1645,100*zp0<=16451]
# initCubeCon = And(10*xp0>=1,100*xp0<=11,10*yp0>=45,100*yp0<=451, 10*zp0>=1645,100*zp0<=16451)


# xp0 = Variable(0)
# yp0 = Variable(1)
# zp0 = Variable(2)
# pd3 = NNC_Polyhedron(3)
# pd3.add_constraint(10*xp0>=1)
# pd3.add_constraint(100*xp0<=11)
# pd3.add_constraint(10*yp0>=45)
# pd3.add_constraint(100*yp0<=451)
# pd3.add_constraint(10*zp0>=1645)
# pd3.add_constraint(100*zp0<=16451)

# midPoints["A"] = [0.1,4.5,164.5]
# currentMidPoint = midPoints["A"]
# currentMidPointString = str(currentMidPoint[0])+"_"+str(currentMidPoint[1])+"_"+str(currentMidPoint[2])
# processedMidPoints[currentMidPointString] = "A"

# initRegionMinMaxValues = [.1,.11,4.5,4.51,164.5,164.51]
# initRegionCornerPoints = [[0.1,4.5,164.5], [0.11,4.5,164.5], [0.1,4.51,164.5], [0.11,4.51,164.5],
#                            [0.1,4.5,164.51], [0.11,4.5,164.51], [0.1,4.51,164.51], [0.11,4.51,164.51]
#                            ]

# regionCornerPoints["A"] = initRegionCornerPoints
# regionMinMaxValues["A"] = initRegionMinMaxValues

# depthOfTheInitialCube = .01


#############Building end here ####################

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


##################original  target region##################
#targetRegionPolyhedron = NNC_Polyhedron(3)
#targetRegionPolyhedron.add_constraint(xp0>=-100)
#targetRegionPolyhedron.add_constraint(xp0<=100)
#targetRegionPolyhedron.add_constraint(yp0>=4)
#targetRegionPolyhedron.add_constraint(yp0<=5)
#targetRegionPolyhedron.add_constraint(zp0>=179)
#targetRegionPolyhedron.add_constraint(zp0<=184)

##################original  target region##################


##################PineTree  target region##################
# targetRegionPolyhedron = NNC_Polyhedron(3)
# targetRegionPolyhedron.add_constraint(xp0>=-100)
# targetRegionPolyhedron.add_constraint(xp0<=100)
# targetRegionPolyhedron.add_constraint(yp0>=0)
# targetRegionPolyhedron.add_constraint(yp0<=50)
# targetRegionPolyhedron.add_constraint(zp0>=159)
# targetRegionPolyhedron.add_constraint(zp0<=169)

##################PineTree  target region end##################

#####Buildings target start here ###########
targetRegionPolyhedron = NNC_Polyhedron(3)
targetRegionPolyhedron.add_constraint(xp0>=-200)
targetRegionPolyhedron.add_constraint(xp0<=200)
targetRegionPolyhedron.add_constraint(yp0>=4)
targetRegionPolyhedron.add_constraint(yp0<=5)
targetRegionPolyhedron.add_constraint(zp0>=120)
targetRegionPolyhedron.add_constraint(zp0<=139)

##############Buildings target end here #########





# vertices = [
        
        
# ]

# nvertices = [
        
# ]
# vertColours = [
        
        
        
# ]       
# tedges = [
        
        
     
# ]

