
from calendar import c
from z3 import *
from pyparma import *
import os
from datetime import date, datetime
import sys
from time import sleep
import anytree

from importlib import reload  # Python 3.4+

import collisionValidityCheck3_p3_2
import environment
import singleTriangleInvRegions30
import floatingpointExpToRational4
import createPoly
# import collisionValidityCheck3
from collections import Counter
import collisionValidityCheck3_p2_2
# import mt_singleTriangleInvRegions23


import sys
sys.path.append('eran_master/tf_verify')


vertices = environment.vertices
nvertices = environment.nvertices


###########collision start###################
def checkForCollision(pathHullConString, currTriangle):
    # print("checkForCollision==>    Checking collision with triangle "+str(currTriangle))
    s1 = Solver()
    set_param('parallel.enable', True)
    set_option(rational_to_decimal=True)
    set_option(precision=20)
    set_param('parallel.enable', True)
    s1.set("sat.local_search_threads", 26)
    s1.set("sat.threads", 26)
    s1.set("timeout", 100)

    xp0, yp0, zp0 = Reals('xp0 yp0 zp0')

    s1.add(simplify(eval(pathHullConString)))

    xk, yk, zk = Reals('xk yk zk')
    u, v, w = Reals('u v w')

    s1.add(u+v+w == 1)
    s1.add(And(u >= 0, v >= 0, w >= 0))

    x0 = vertices[nvertices[currTriangle*3+0]*3+0]
    y0 = vertices[nvertices[currTriangle*3+0]*3+1]
    z0 = vertices[nvertices[currTriangle*3+0]*3+2]

    x1 = vertices[nvertices[currTriangle*3+1]*3+0]
    y1 = vertices[nvertices[currTriangle*3+1]*3+1]
    z1 = vertices[nvertices[currTriangle*3+1]*3+2]

    x2 = vertices[nvertices[currTriangle*3+2]*3+0]
    y2 = vertices[nvertices[currTriangle*3+2]*3+1]
    z2 = vertices[nvertices[currTriangle*3+2]*3+2]

    # print(x0, y0, z0, x1, y1, z1, x2, y2, z2)

    s1.add(xk == (u*x0+v*x1+w*x2))
    s1.add(yk == (u*y0+v*y1+w*y2))
    s1.add(zk == (u*z0+v*z1+w*z2))

    s1.add(xp0 == xk)
    s1.add(yp0 == yk)
    s1.add(zp0 == zk)

    # print(s1.check())

    # while (True):
    solverResult = s1.check()
    if(solverResult == sat):
        del(s1)
        print("collision detected. On the path ")
        return 1
    elif(solverResult == unsat):
        del(s1)
        return 0
    del(s1)
    # print("timeout while checking collision of triangle"+str(currTriangle))
    # print("retrying... ")
    checkForCollision(pathHullConString, currTriangle)



def errorTriangleCheck(t):
    currTriangle = t
    vertex0 = nvertices[currTriangle*3+0]
    vertex1 = nvertices[currTriangle*3+1]
    vertex2 = nvertices[currTriangle*3+2]
    currTriangleVertices = [vertex0, vertex1,vertex2]

    v0Vertex = [vertices[currTriangleVertices[0]*3+0], vertices[currTriangleVertices[0]*3+1],vertices[currTriangleVertices[0]*3+2] ]
    v1Vertex = [vertices[currTriangleVertices[1]*3+0], vertices[currTriangleVertices[1]*3+1],vertices[currTriangleVertices[1]*3+2] ]
    v2Vertex = [vertices[currTriangleVertices[2]*3+0], vertices[currTriangleVertices[2]*3+1],vertices[currTriangleVertices[2]*3+2] ]


    if Counter(v0Vertex) == Counter(v1Vertex) or Counter(v0Vertex) == Counter(v2Vertex) or Counter(v1Vertex) == Counter(v2Vertex):
        # print(v0Vertex, v1Vertex, v2Vertex)
        # print("Error triangle")
        # print(currTriangle)
        return 1
    else:
        return 0





programStartTime = datetime.now()
print("Program Started..")
print(str(datetime.now()))

os.remove("ErrorLog.txt")
with open("ErrorLog.txt", "w") as f:
    f.write(str(datetime.now())+"\n\n")

eFile = open("collisionData.txt","w")
eFile.write("Program Started project2 "+(str(datetime.now())+"\n"))
eFile.write("number of edges = "+str(environment.numOfEdges))
eFile.write(str(environment.initCubeCon))
eFile.write(str(environment.targetRegionPolyhedron.minimized_constraints()))
eFile.close()

lFile = open("log.txt","w")
lFile.write("Program Started @ : "+(str(datetime.now())+"\n"))
lFile.write("\nNumber of edges = "+str(environment.numOfEdges))
lFile.write("\nInitial region cons: "+str(environment.initCubeCon))
lFile.write("\nTarget region cons: "+str(environment.targetRegionPolyhedron.minimized_constraints()))
lFile.close()

currAbsGroupName = "A_"
currAbsGroupRegionCons = environment.initCubeCon



singleTriangleInvRegions30.computePixelIntervals(
    currAbsGroupName, currAbsGroupRegionCons)

# print("time Taken = ", datetime.now() - programStartTime)


# mt_singleTriangleInvRegions23.computePixelIntervals(currAbsGroupName, currAbsGroupRegionCons)

# currentMidPoint = environment.midPoints["A"]
# currentMidPointString = str(currentMidPoint[0])+"_"+str(currentMidPoint[1])+"_"+str(currentMidPoint[2])
# environment.processedMidPoints[currentMidPointString] = 1

loopCount = 1
targetReachCount = 0
alreadyProcessedCount = 0

# for pre, fill, node in anytree.RenderTree(environment.A):
#     print("%s%s" % (pre, node.name))


# print("current stack : ", environment.absStack)

whileLoopCount = 1
while (environment.absStack):
   
    # print("while loop count = ", whileLoopCount)
    for pre, fill, node in anytree.RenderTree(environment.A):
        print("%s%s" % (pre, node.name))
    
    whileLoopCount += 1

    currGroup = environment.absStack.pop()
    print("currstacktop = ", currGroup)

    currDnnOutput = int(currGroup[-1:])
    currGroupName = currGroup[0:currGroup.rfind("_")]
    currGroupCons = environment.groupCubeZ3[currGroupName]

    # print("currGroupName = ",currGroupName)
    # print("currDnnOutput = ", currDnnOutput)
    # print("currGroupCons = ", currGroupCons)

    currMidPoint = environment.midPoints[currGroupName]

    nextMidPoint = [0, 0, 0]
    nextMidPoint[0] = currMidPoint[0]
    nextMidPoint[1] = currMidPoint[1]
    nextMidPoint[2] = currMidPoint[2]

    # exit()

    # if(whileLoopCount == 5):
    #     print("forth loop sleeping")
    #     sleep(100)
    # sleep(2)

    currGroupCubeCons = environment.groupCube[currGroupName]
    currRegionPolyhedron = NNC_Polyhedron(3)
    currRegionPolyhedron.add_constraints(currGroupCubeCons)

    # print(environment.targetRegionPolyhedron.minimized_constraints())
    # print("\n\n")
    # print(currRegionPolyhedron.minimized_constraints())

    currRegionPolyhedron.intersection_assign(
        environment.targetRegionPolyhedron)
    # print("intersection polyhedron cons ")
    # print(currRegionPolyhedron.minimized_constraints())
    # print(currRegionPolyhedron.is_empty())

    if(environment.targetRegionPolyhedron.is_disjoint_from(currRegionPolyhedron)):
        # print("\n")
        pass
    else:
        # print("target reached ==> " + str(targetReachCount))
        targetReachCount += 1
        if(environment.targetRegionPolyhedron.contains(currRegionPolyhedron)):
            # print("fully inside ")
            # sleep(2)
            continue
        else:

            # print("Partially reached target")

            currRegionPolyhedron.poly_difference_assign(
                environment.targetRegionPolyhedron)
            currGroupCubeCons = currRegionPolyhedron.minimized_constraints()
            # TODO update code to take the difference

    newFormula1 = ""
    xp0, yp0, zp0 = Reals('xp0 yp0 zp0')
    xp1, yp1, zp1 = Reals('xp1 yp1 zp1')
    if(currDnnOutput == 0):
        # print("currDnnOutput = ", currDnnOutput)
        newFormula1 = Exists([xp0, yp0, zp0], And(
            currGroupCons, xp1 == xp0-.5, yp1 == yp0, zp1 == zp0-.866))
        nextMidPoint[0] = currMidPoint[0] - 0.5
        nextMidPoint[2] = currMidPoint[2] - 0.866

    elif(currDnnOutput == 1):
        # print("currDnnOutput = ", currDnnOutput)
        newFormula1 = Exists([xp0, yp0, zp0], And(
            currGroupCons, xp1 == xp0, yp1 == yp0, zp1 == zp0-1))
        nextMidPoint[2] = currMidPoint[2] - 1
    elif(currDnnOutput == 2):
        # print("currDnnOutput = ", currDnnOutput)
        newFormula1 = Exists([xp0, yp0, zp0], And(
            currGroupCons, xp1 == xp0+.5, yp1 == yp0, zp1 == zp0-.866))
        nextMidPoint[0] = currMidPoint[0] + 0.5
        nextMidPoint[2] = currMidPoint[2] - 0.866

    nextMidPointString = str(
        nextMidPoint[0])+"_"+str(nextMidPoint[1])+"_"+str(nextMidPoint[2])

    # print("new formula --->")
    # print(newFormula1)
    # sleep(2)

    set_option(rational_to_decimal=False)
    set_option(precision=10)
    g = Goal()
    g.add((newFormula1))

    t1 = Tactic('simplify')
    t2 = Tactic('qe')
    t = Then(t2, t1)
    # print (t(g))

    # print("\n\n converting to PPL expression")
    oldExp = t(g)[0]
    # print(oldExp)
    updatedExpString = []

    for n in range(0, len(t(g)[0])):
        exp = str(t(g)[0][n])
        # print(exp)
        exp = exp.replace("xp1", "xp0")
        exp = exp.replace("yp1", "yp0")
        exp = exp.replace("zp1", "zp0")
        exp = exp.replace("\n", "")

        try:
            updatedExpString.append(eval(exp))
        except:
            # print("exception handled")
            exp = exp.replace("/", "//")
            updatedExpString.append(eval(exp))

    updateExp = []

    # print("\n\n")
    for n in range(0, len(t(g)[0])):
        exp = t(g)[0][n]
        # print("current expression to conversion")
        # print(exp)

        # exp = str(exp).replace("xp0","xp1")
        # exp = str(exp).replace("yp0","yp1")
        # exp = str(exp).replace("zp0","zp1")
        # exp = str(exp).replace("\n", "")
        # print(exp)
        # print("\n\n")

        try:
            exp = eval(str(exp).replace("\n", ""))
        except:
            
            exit(0)

        newExp = floatingpointExpToRational4.converteToPPLExpression(exp)
        newExp = str(newExp)

        newExp = newExp.replace("xp1", "xp0")
        newExp = newExp.replace("yp1", "yp0")
        newExp = newExp.replace("zp1", "zp0")
        newExp = newExp.replace("\n", "")
        # print("\n\n")
        # print("returned expression ")
        # print(newExp)
        updateExp.append(newExp)
    # print("\n\n")
    # print("oldExp = ",oldExp)
    # print("updateExp = ",updateExp)

    pd4 = NNC_Polyhedron(3)
    xp0 = Variable(0)
    yp0 = Variable(1)
    zp0 = Variable(2)

    conFile = open("createPoly.py", "w")
    tempstring = "from pyparma import *\n\ndef getPoly():\n    xp0 = Variable(0)\n\
    yp0 = Variable(1)\n\
    zp0 = Variable(2)\n\
    pd3 = NNC_Polyhedron(3)\n"
    for n in range(0, len(updateExp)):
        tempstring += "    pd3.add_constraint(" + \
            str(updateExp[n]).replace("?", "")+")\n"

    tempstring += "    return pd3\n"
    conFile.write(tempstring)

    conFile.close()
    createPoly = reload(createPoly)
    pd4 = createPoly.getPoly()

    # environment.groupCube[currGroup] = pd4.minimized_constraints()
    # print("current GRoup name ",currGroupName)
    # print("currentGroupregion =;")
    # # print(currGroup+"_"+str(currDnnOutput))
    # print(pd4.minimized_constraints())
    # environment.groupCube[currGroup+"_"+str(currGroupDnnOutput)]  = pd4.minimized_constraints()
    # environment.groupCube[currGroup+"_0"]  = pd4.minimized_constraints()
    # environment.groupCube[currGroup+"_1"]  = pd4.minimized_constraints()
    # environment.groupCube[currGroup+"_2"]  = pd4.minimized_constraints()
    environment.groupCube[currGroup] = pd4.minimized_constraints()

    environment.groupCubePostRegion[currGroup] = pd4.minimized_constraints()

    ########################collision check start####################

    # print("\n\n>>>>>>>>>>>>>>>>>>>>collision check started<<<<<<<<<<<<<<<<<<<\n\n")
    # sleep(5)

    currGroupRegionCons = environment.groupCube[currGroupName]

    pd5 = NNC_Polyhedron(3)
    pd5.add_constraints(currGroupRegionCons)

    # print("current region cons :", pd5.minimized_constraints())
    # print("next region cons : ", pd4.minimized_constraints())

    pd5.poly_hull_assign(pd4)

    # print("path hull cons ", pd5.minimized_constraints())
    # sleep(5)
    pathHullConString = pd5.minimized_constraints()

    pathHullConString = str(pathHullConString)
    pathHullConString = pathHullConString.replace("x0", "xp0")
    pathHullConString = pathHullConString.replace("x1", "yp0")
    pathHullConString = pathHullConString.replace("x2", "zp0")
    pathHullConString = pathHullConString.replace(" = ", "==")
    pathHullConString = pathHullConString.replace("Constraint_System {", " ")
    pathHullConString = pathHullConString.replace("}", " ")
    pathHullConString = "And("+str(pathHullConString)+")"

    # print("\n after replacing path hull cons\n")
    # print("\n\n",pathHullConString)

    # print("\n\n")
    #global collisionFlag
    #collisionFlag = 0
        #collisionFlag = 0
    print("time Taken = ", datetime.now() - programStartTime)

    intersectionRegionEqualsToRegion = 0
    for t in range(0, environment.numOfTriangles):
        # print("\n\n\nfrom main function")
        # print("main ==> checking collision of pathHull with the triangle "+str(t))
        
        
        x0 = vertices[nvertices[t*3+0]*3+0]
        # y0 = vertices[nvertices[currTriangle*3+0]*3+1]
        z0 = vertices[nvertices[t*3+0]*3+2]

        x1 = vertices[nvertices[t*3+1]*3+0]
        # y1 = vertices[nvertices[currTriangle*3+1]*3+1]
        z1 = vertices[nvertices[t*3+1]*3+2]

        x2 = vertices[nvertices[t*3+2]*3+0]
        # y2 = vertices[nvertices[currTriangle*3+2]*3+1]
        z2 = vertices[nvertices[t*3+2]*3+2]

        
        #if all z values are less than or greater than the hull then skip the triangle
        tempCurrMinZ = currMidPoint[2] + environment.depthOfTheInitialCube
        tempCurrMaxZ = nextMidPoint[2] - environment.depthOfTheInitialCube
        if ((z0 > tempCurrMinZ and z1 > tempCurrMinZ and z2 > tempCurrMinZ) or (z0 < tempCurrMaxZ and z1 < tempCurrMaxZ and z2 < tempCurrMaxZ)):
            # print("skipping the triangle "+str(t))
            continue

        
        if errorTriangleCheck(t) ==1:
            continue
        
        collision = checkForCollision(pathHullConString, t)
        if collision == 1:
            # print(
            #     "main ==> collision detected and checking for validity!!!!!!!!!!!!!!!!!!!!!!!1")
            # print("intersectionRegionEqualsToRegion ", intersectionRegionEqualsToRegion)
            # print(datetime.now())
            # print("time Taken = ", datetime.now() - programStartTime)
            # sleep(2)

            eFile = open("collisionData.txt","a")
            eFile.write("Collision detected "+str(t)+"\n")
            eFile.write("time Taken = "+str(datetime.now() - programStartTime)+"\n")
            eFile.write("currGroupName = "+str(currGroupName)+"\n")
            eFile.write("currDnnOutput = "+str(currDnnOutput))
            eFile.write("triangle = "+str(t)+"\n")
            eFile.write("pathHullConString = "+str(pathHullConString)+"\n")
            eFile.close()
            colStatus = 0

           

############TOREMOVE####################################################################################
           

            # colStatus = collisionValidityCheck3_p3_1.checkValidityOfCollision(
            #      currGroupName, t, currGroupRegionCons, currDnnOutput, pd5)

            # print("intersectionRegionEqualsToRegion ", intersectionRegionEqualsToRegion)
            returnIntersectRegion = And(True)
            if intersectionRegionEqualsToRegion == 0:
                #with multiprocessing
                colStatus, returnIntersectRegion = collisionValidityCheck3_p3_2.checkValidityOfCollision(
                    currGroupName, t, currGroupRegionCons, currDnnOutput, pd5)
                # #continue
                # print("colStatus = ", colStatus)
                # print("returnIntersectRegion ", returnIntersectRegion)
                # print("currGroupCons ", currGroupCons)
                # Create solvers to check equivalence
                xp0, yp0, zp0 = Reals('xp0 yp0 zp0')
                si1 = Solver()
                si2 = Solver()

                # Logical equivalence: set1 implies set2
                si1.add(And(returnIntersectRegion , Not(And(currGroupCons))))
                # Logical equivalence: set2 implies set1
                si2.add(And(currGroupCons , Not(And(returnIntersectRegion))))

                # Check equivalence
                equivalent = (si1.check() == unsat) and (si2.check() == unsat)   

                if equivalent:
                    print("The two sets of constraints are equivalent.")
                    intersectionRegionEqualsToRegion = 1
                else:
                    print("The two sets of constraints are not equivalent.")

               
            xp0 = Variable(0)
            yp0 = Variable(1)
            zp0 = Variable(2)

            if(colStatus == 1):
                print("Collision is valid")
                print("TRUE COLLISION DETECTED, CHECK collisionData.txt")
                collisionFile = open("collisionData.txt", "a")
                collisionFile.write("Time taken")
                collisionFile.write(str(datetime.now() - programStartTime))
                collisionFile.write("triangle = ")
                collisionFile.write(str(t))
                collisionFile.write("currGroupName")
                collisionFile.write(str(currGroupName))
                print("Details written to collisionData.txt file")
                
                
                lFile = open("log.txt","a")
                lFile.write("\n\nProgram Finished, Environment is UNSAFE \n")
                lFile.write("time Taken = "+str(datetime.now() - programStartTime))
                lFile.write("\nNumber of spurious collisions = "+str(environment.spuriousCollisionCount))
                lFile.write("\nNumber of regions interval image computed = "+str(whileLoopCount-targetReachCount-alreadyProcessedCount))
                lFile.write("\nNumber of target reach = "+str(targetReachCount))
                lFile.close()

                exit(0)
            elif(colStatus == 0):
                # print("True  >>>>>>>>>>>> Spurios Collision<<<<<<<<09809<<<<")
                # print("Program continue......")
                environment.spuriousCollisionCount += 1
                eFile = open("collisionData.txt","a")
                eFile.write("True  >>>>>>>>>>>> Spurios Collision<<<<<<<<09809<<<<")
                eFile.write("time Taken = "+str(datetime.now() - programStartTime)+"\n")
               
                eFile.close()

                # add the spurious collision details to the tree,
                # it may be a valid collision in future
                tempCurrGroupName = currGroupName
                while tempCurrGroupName != "A":
                    
                    if tempCurrGroupName in environment.spuriousCollisionData:
                        environment.spuriousCollisionData[tempCurrGroupName].append(
                            [currGroupName, str(t), currDnnOutput])
                    else:
                        environment.spuriousCollisionData[tempCurrGroupName] = [
                            [currGroupName, str(t), currDnnOutput]]
                    tempCurrGroupName = tempCurrGroupName[:tempCurrGroupName.rfind(
                        "_")]

                sleep(2)

            # exit(0)
                continue

            
        else:           
            pass

  
    

    nextGroupName = currGroup+"_"
    
    if nextMidPointString in environment.processedMidPoints:
        
        currNextNode = environment.processedMidPoints[nextMidPointString]
        alreadyProcessedCount +=1
        
        if currNextNode in environment.spuriousCollisionData:
            environment.groupCube[currGroup] = environment.groupCube[currNextNode]
            environment.groupCubePostRegion[currGroup] = environment.groupCubePostRegion[currNextNode]
            environment.groupCubeZ3[currGroup] = environment.groupCubeZ3[currNextNode]

            spuriousCollisions = environment.spuriousCollisionData[currNextNode]
           

            if len(spuriousCollisions) >0:                
                continue
            for spc in range(0, len(spuriousCollisions)):
                print("\n\n current spurious collision = ",
                      spuriousCollisions[spc])
                spcPath = spuriousCollisions[spc][0]
                spcObstacle = spuriousCollisions[spc][1]
                spcDnnOutput = spuriousCollisions[spc][2]
                print("spcPath = ", spcPath)
                print("spcObstacle = ", spcObstacle)
                print("spcDnnOutput = ", spcDnnOutput)
                pathToExplore = spcPath[len(currNextNode)+1:]
                print("pathToExplore = ", pathToExplore)

                pathToAddCons = currGroup + "_" + pathToExplore
                pathToCheckCollision = currGroup + "_" + pathToExplore
                print("pathToAddCons = ", pathToAddCons)
                orginalConsPath = currNextNode + "_" + pathToExplore
                print("orginalConsPath = ", orginalConsPath)

                environment.groupCube[pathToAddCons] = environment.groupCube[orginalConsPath]
                environment.groupCubePostRegion[pathToAddCons] = environment.groupCubePostRegion[orginalConsPath]
                environment.groupCubeZ3[pathToAddCons] = environment.groupCubeZ3[orginalConsPath]

                for lensubPath in range(0, pathToExplore.count("_")):
                    substrings = pathToExplore.split("_", lensubPath+1)
                    # return char.join(substrings[:-1])
                    print("substrings = ", substrings)
                    pathSegmentToAttach = "_".join(substrings[:-1])
                    print("pathSegmentToAttach = ", pathSegmentToAttach)

                    pathToAddCons = currGroup + "_" + pathSegmentToAttach
                    print("pathToAddCons = ", pathToAddCons)

                    orginalConsPath = currNextNode + "_" + pathSegmentToAttach
                    print("orginalConsPath = ", orginalConsPath)

                    environment.groupCube[pathToAddCons] = environment.groupCube[orginalConsPath]
                    environment.groupCubePostRegion[pathToAddCons] = environment.groupCubePostRegion[orginalConsPath]
                    environment.groupCubeZ3[pathToAddCons] = environment.groupCubeZ3[orginalConsPath]

                collCurrRegionCons = environment.groupCube[currGroup +
                                                           "_" + pathToExplore]
                collNextRegionCons = environment.groupCube[currNextNode +
                                                           "_" + pathToExplore + "_" + str(spcDnnOutput)]

                pd6 = NNC_Polyhedron(3)
                pd6.add_constraints(collCurrRegionCons)

                pd7 = NNC_Polyhedron(3)
                pd7.add_constraints(collNextRegionCons)

                

                pd7.poly_hull_assign(pd6)

                colStatus = collisionValidityCheck3_p3_2.checkValidityOfCollision(pathToCheckCollision, int(spcObstacle),
                                                                             environment.groupCubeZ3[pathToCheckCollision], spcDnnOutput, pd7)
                # # checkCollisionValidity(currGroup,t)

                if(colStatus == 1):
                    print("Collision is valid")
                    print("TRUE COLLISION DETECTED, CHECK collisionData.txt")
                    collisionFile = open("collisionData.txt", "a")
                    collisionFile.write("Time taken")
                    collisionFile.write(str(datetime.now() - programStartTime))
                    collisionFile.write("triangle = ")
                    collisionFile.write(str(t))
                    collisionFile.write("currGroupName")
                    collisionFile.write(str(currGroup + "_" + pathToExplore))
                    
                    
                    lFile = open("log.txt","a")
                    lFile.write("\n\nProgram Finished, Environment is UNSAFE \n")
                    lFile.write("\ntime Taken = "+str(datetime.now() - programStartTime))
                    lFile.write("\nNumber of spurious collisions = "+str(environment.spuriousCollisionCount))
                    lFile.write("\nNumber of regions interval image computed = "+str(whileLoopCount+1-targetReachCount-alreadyProcessedCount))
                    lFile.write("\nNumber of target reach = "+str(targetReachCount))
                    lFile.close()
                    sleep(1)
                    exit(0)
                elif(colStatus == 0):
                    # print("True  >>>>>>>>>>>> Spurios Collision<<<<<<<<<<<<")
                    # print("Program continue......")
                    

                    # add the spurious collision details to the tree,
                    # it may be a valid collision in future
                    tempCurrGroupName = currGroup + "_" + pathToExplore
                    while tempCurrGroupName != "A":
                        print("tempCurrGroupName = ", tempCurrGroupName)
                        if tempCurrGroupName in environment.spuriousCollisionData:
                            environment.spuriousCollisionData[tempCurrGroupName].append(
                                [currGroupName, str(t), spcDnnOutput])
                        else:
                            environment.spuriousCollisionData[tempCurrGroupName] = [
                                [currGroupName, str(t), spcDnnOutput]]
                        tempCurrGroupName = tempCurrGroupName[:tempCurrGroupName.rfind(
                            "_")]

        continue
    else:
        environment.processedMidPoints[nextMidPointString] = str(currGroup)
        environment.midPoints[str(currGroup)] = nextMidPoint
       

    if intersectionRegionEqualsToRegion == 1:
        continue
    
    tempFromula = And(True)
    for n in range(0, len(updatedExpString)):
        tempFromula = And(tempFromula, (updatedExpString[n]))

    
    environment.groupCubeZ3[currGroup] = tempFromula
    
    singleTriangleInvRegions30.computePixelIntervals(
        nextGroupName, tempFromula)

    loopCount = loopCount+1
   


print("\n\nProgram Finished, Environment is SAFE")
print(datetime.now())
print("time Taken = ", datetime.now() - programStartTime)


lFile = open("log.txt","a")
lFile.write("\n\nProgram Finished, Environment is SAFE \n")
lFile.write("\ntime Taken = "+str(datetime.now() - programStartTime))
lFile.write("\nNumber of spurious collisions = "+str(environment.spuriousCollisionCount))
lFile.write("\nNumber of regions interval image computed = "+str(whileLoopCount-targetReachCount-alreadyProcessedCount))
lFile.write("\nNumber of target reach = "+str(targetReachCount))
lFile.close()


# sleep(10)
