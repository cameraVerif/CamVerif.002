#libraries
from datetime import datetime
from z3 import *
from pyparma import *
import os
import sys
from time import sleep
import anytree
from importlib import reload  # Python 3.4+
from collections import Counter


import environment
import intervalImageP3_4
import floatingpointExpToRational4
import createPoly
import collisionValidityCheck3_p3_2


vertices = environment.vertices
nvertices = environment.nvertices


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
    # print(solverResult)
    if(solverResult == sat):       
        # print("collision detected. On the path ,\n value of intersection")
        m=s1.model()
        # print(m)
        # lcount =1
        # while(s1.check() == sat):
        #     print("while loop started : ", lcount)
        #     m=s1.model()
        #     print(m)    
        #     print("model generated")
        #     notThePointCons = And(xp0 != m[xp0], yp0 != m[yp0], zp0 != m[zp0])
        #     s1.add(notThePointCons)
        #     print("notThePointCons added")
        #     lcount += 1
            

        # del(s1)
        return 1
    elif(solverResult == unsat):
        # del(s1)
        return 0
    # del(s1)
    print("timeout while checking collision of triangle"+str(currTriangle))
    print("retrying... ")
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
        print(v0Vertex, v1Vertex, v2Vertex)
        print("Error triangle")
        print(currTriangle)
        return 1
    else:
        return 0



programStartTime = datetime.now()
print("From main_P3_2: Program Started..")
print(str(datetime.now()))

print("number of edges = ",environment.numOfEdges)

eFile = open("ErrorLog.txt","w")
eFile.write("Program Started "+(str(datetime.now())+"\n"))
eFile.close()

eFile = open("collisionData.txt","w")
eFile.write("Program Started "+(str(datetime.now())+"\n"))
eFile.write("number of edges = "+str(environment.numOfEdges))
eFile.write(str(environment.initCubeCon))
eFile.close()

lFile = open("log.txt","w")
lFile.write("Program Started @ : "+(str(datetime.now())+"\n"))
lFile.write("\nNumber of edges = "+str(environment.numOfEdges))
lFile.write("\nInitial region cons: "+str(environment.initCubeCon))
lFile.write("\nTarget region cons: "+str(environment.targetRegionPolyhedron.minimized_constraints()))
lFile.close()


currAbsGroupName = "A_"
currAbsGroupRegionCons = environment.initCubeCon
currRegionMinMaxValues = environment.initRegionMinMaxValues
currRegionCornerPoints = environment.initRegionCornerPoints

intervalImageP3_4.computeIntervalImage_P3(currAbsGroupName, currAbsGroupRegionCons,
                                           currRegionMinMaxValues, currRegionCornerPoints)

# print("time Taken = ", datetime.now() - programStartTime)


loopCount = 0
targetReachCount = 0
spuriousCollisionCount = 0
alreadyProcessedCount = 0

for pre, fill, node in anytree.RenderTree(environment.A):
    print("%s%s" % (pre, node.name))


# print("current stack : ", environment.absStack)

whileLoopCount = 0





while (environment.absStack):

    # print("\n\n\n while loop new ",datetime.now() - programStartTime)
    # print("time Taken = ", datetime.now() - programStartTime)
    # print(str(datetime.now()))
    # now = datetime.now()
    # print(str(now))
    if whileLoopCount %10 == 0:
        print("current stack : ", environment.absStack)
        print("time Taken = ", datetime.now() - programStartTime)
    if whileLoopCount %2000 == 0:
        for pre, fill, node in anytree.RenderTree(environment.A):
            print("%s%s" % (pre, node.name))
   
    whileLoopCount += 1

    currGroup = environment.absStack.pop()
    
    

    currDnnOutput = int(currGroup[-1:])
    currGroupName = currGroup[0:currGroup.rfind("_")]
    currGroupCons = environment.groupCubeZ3[currGroupName]

    # print("currGroupName = ",currGroupName)
    # print("currDnnOutput = ", currDnnOutput)
    # print("currGroupCons = ", currGroupCons)

    
    currMidPoint = environment.midPoints[currGroupName]
    currRegionCornerPoints = environment.regionCornerPoints[currGroupName]
    currRegionMinMaxValues = environment.regionMinMaxValues[currGroupName]

    # print("currRegionCornerPoints = ", currRegionCornerPoints)
    # print("currRegionMinMaxValues = ", currRegionMinMaxValues)
    

    nextRegionCornerPoints = []
    nextRegionMinMaxValues = []

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
        # print("target not reached")
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
            
    # initRegionMinMaxValues = [0.1,0.101,4.5,4.501,121.5,121.501]
    # initRegionCornerPoints = [[0.1,4.5,121.5], [0.101,4.5,121.5], [0.1,4.501,121.5], [0.101,4.501,121.5],
    #                       [0.1,4.5,121.501], [0.101,4.5,121.501], [0.1,4.501,121.501], [0.101,4.501,121.501]
    #                       ]

    newFormula1 = ""
    xp0, yp0, zp0 = Reals('xp0 yp0 zp0')
    xp1, yp1, zp1 = Reals('xp1 yp1 zp1')
    if(currDnnOutput == 0):
        # print("currDnnOutput = ", currDnnOutput)
        newFormula1 = Exists([xp0, yp0, zp0], And(
            currGroupCons, xp1 == xp0-.5, yp1 == yp0, zp1 == zp0-.866))
        nextMidPoint[0] = currMidPoint[0] - 0.5
        nextMidPoint[2] = currMidPoint[2] - 0.866


        for currPoint in currRegionCornerPoints:
            nextPoint = list(currPoint)
            nextPoint[0] = currPoint[0] - 0.5
            nextPoint[2] = currPoint[2] - 0.866
            nextRegionCornerPoints.append(nextPoint)
        
        nextRegionMinMaxValues = list(currRegionMinMaxValues)
        nextRegionMinMaxValues[0] = currRegionMinMaxValues[0] -  0.5
        nextRegionMinMaxValues[1] = currRegionMinMaxValues[1] -  0.5

        nextRegionMinMaxValues[4] = currRegionMinMaxValues[4] -  0.866
        nextRegionMinMaxValues[5] = currRegionMinMaxValues[5] -  0.866



    elif(currDnnOutput == 1):
        # print("currDnnOutput = ", currDnnOutput)
        newFormula1 = Exists([xp0, yp0, zp0], And(
            currGroupCons, xp1 == xp0, yp1 == yp0, zp1 == zp0-1))
        nextMidPoint[2] = currMidPoint[2] - 1

        for currPoint in currRegionCornerPoints:
            nextPoint = list(currPoint)
            nextPoint[2] = currPoint[2] - 1
            nextRegionCornerPoints.append(nextPoint)
        
        nextRegionMinMaxValues = list(currRegionMinMaxValues)
        nextRegionMinMaxValues[4] = currRegionMinMaxValues[4] - 1
        nextRegionMinMaxValues[5] = currRegionMinMaxValues[5] - 1


    elif(currDnnOutput == 2):
        # print("currDnnOutput = ", currDnnOutput)
        newFormula1 = Exists([xp0, yp0, zp0], And(
            currGroupCons, xp1 == xp0+.5, yp1 == yp0, zp1 == zp0-.866))
        nextMidPoint[0] = currMidPoint[0] + 0.5
        nextMidPoint[2] = currMidPoint[2] - 0.866


        for currPoint in currRegionCornerPoints:
            # print("CurrPoint = ", currPoint)
            nextPoint = list(currPoint)
            nextPoint[0] = currPoint[0] + 0.5
            nextPoint[2] = currPoint[2] - 0.866
            # print("nextPoint = ", nextPoint)
            nextRegionCornerPoints.append(nextPoint)
        
        nextRegionMinMaxValues = list(currRegionMinMaxValues)
        nextRegionMinMaxValues[0] = currRegionMinMaxValues[0] +  0.5
        nextRegionMinMaxValues[1] = currRegionMinMaxValues[1] +  0.5



        nextRegionMinMaxValues[4] = currRegionMinMaxValues[4] -  0.866
        nextRegionMinMaxValues[5] = currRegionMinMaxValues[5] -  0.866


    environment.regionCornerPoints[currGroup] = nextRegionCornerPoints
    environment.regionMinMaxValues[currGroup] = nextRegionMinMaxValues

    nextMidPointString = str(
        nextMidPoint[0])+"_"+str(nextMidPoint[1])+"_"+str(nextMidPoint[2])


    # print("currRegionCornerPoints = ", currRegionCornerPoints)
    # print("nextRegionCornerPoints = ", nextRegionCornerPoints)

    # print("\n\n")
    # print("currRegionMinMaxValues = ", currRegionMinMaxValues)
    # print("nextRegionMinMaxValues = ", nextRegionMinMaxValues)

    

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
            # print("exception handled2 main_abs_1 @108")
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
    # print("time Taken = ", datetime.now() - programStartTime)

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
            # exit()
            # print("\n\nValidity check started\n\n")
            # print(x0, z0, x1, z1, x2, z2)
            # print("pd4.minimized_constraints() = ", pd4.minimized_constraints())
            # print("pd5.minimized_constraints() = ", pd5.minimized_constraints())
            # print("pathHullConString = ", pathHullConString)

          

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
                    # print("The two sets of constraints are equivalent.")
                    intersectionRegionEqualsToRegion = 1
                else:
                    # print("The two sets of constraints are not equivalent.")
                    pass

                # sleep(5)
            xp0 = Variable(0)
            yp0 = Variable(1)
            zp0 = Variable(2)

            if(colStatus == 1):
                print("Collision is valid")
                print("TRUE COLLISION DETECTED, CHECK collisionData.txt")
                print("Details written to collisionData.txt file")
                collisionFile = open("collisionData.txt", "a")
                collisionFile.write("Time taken")
                collisionFile.write(str(datetime.now() - programStartTime))
                collisionFile.write("triangle = ")
                collisionFile.write(str(t))
                collisionFile.write("currGroupName")
                collisionFile.write(str(currGroupName))
                # print("Details written to collisionData.txt file")
                # print("time Taken = ", datetime.now() - programStartTime)
                # for pre, fill, node in anytree.RenderTree(environment.A):
                #     print("%s%s" % (pre, node.name))

                # sleep(10)

                lFile = open("log.txt","a")
                lFile.write("\n\nProgram Finished, Environment is UNSAFE \n")
                lFile.write("time Taken = "+str(datetime.now() - programStartTime))
                lFile.write("\nNumber of spurious collisions = "+str(spuriousCollisionCount))
                lFile.write("\nNumber of regions interval image computed = "+str(whileLoopCount-targetReachCount-alreadyProcessedCount))
                lFile.write("\nNumber of target reach = "+str(targetReachCount))
                lFile.close()
                exit(0)
            elif(colStatus == 0):
                # print("True  >>>>>>>>>>>> Spurios Collision<<<<<<<<09809<<<<")
                # print("Program continue......")
                
                eFile = open("collisionData.txt","a")
                eFile.write("True  >>>>>>>>>>>> Spurios Collision<<<<<<<<09809<<<<")
                eFile.write("time Taken = "+str(datetime.now() - programStartTime)+"\n")
               
                eFile.close()

                spuriousCollisionCount += 1

                # add the spurious collision details to the tree,
                # it may be a valid collision in future
                tempCurrGroupName = currGroupName
                while tempCurrGroupName != "A":
                    # print("tempCurrGroupName = ", tempCurrGroupName)
                    if tempCurrGroupName in environment.spuriousCollisionData:
                        environment.spuriousCollisionData[tempCurrGroupName].append(
                            [currGroupName, str(t), currDnnOutput])
                    else:
                        environment.spuriousCollisionData[tempCurrGroupName] = [
                            [currGroupName, str(t), currDnnOutput]]
                    tempCurrGroupName = tempCurrGroupName[:tempCurrGroupName.rfind(
                        "_")]

                # sleep(2)

            # exit(0)
                continue

            # sleep(1000000)
            # print("exiting")
            # exit(0)
            ####
            # if valid collision then set flag =1
            # if (result == 1):
            #   collisionFlag =1
            #
            #
            #
        # else:
        #     print("main ==> No collision with the triangle "+str(t))

    # if(collisionFlag == 1):
    #    collisionFlag = 0
    #    continue

    # print("main ==> collision check finished")
    # # print(str(currGroupName))
    # print("time Taken = ", datetime.now() - programStartTime)
    # sleep(4)
    ########################collision check end#############

    # exit()
    

    nextGroupName = currGroup+"_"
    # # print("nextGroupName = ", nextGroupName)

    # # if region already processed then skip to next region
    # # print("MidPoints = ", environment.midPoints)
    # # print("processedMidPoints = ", environment.processedMidPoints)
    # # print("nextMidPoint = ", nextMidPoint)
    # # print("currGroup = ", currGroup)
    # # print(nextMidPointString)
    if nextMidPointString in environment.processedMidPoints:
        # print("region already processed")
        alreadyProcessedCount += 1
        # check whether the region has a spurious collision ahedad or not
        # if there is spurious collision then find out the region where the collision happened
        currNextNode = environment.processedMidPoints[nextMidPointString]
        
        
        if currNextNode in environment.spuriousCollisionData:
            # there is a spurious collision ahead that is previosuly detected
            # find the path to that region and process it
            # there may be multiple spurious collisions ahead
            # so process all of them
            environment.groupCube[currGroup] = environment.groupCube[currNextNode]
            environment.groupCubePostRegion[currGroup] = environment.groupCubePostRegion[currNextNode]
            environment.groupCubeZ3[currGroup] = environment.groupCubeZ3[currNextNode]

            spuriousCollisions = environment.spuriousCollisionData[currNextNode]
            # print("number of spurious collisions = ", len(spuriousCollisions))
            # print("spuriousCollisions = ", spuriousCollisions)

            # if len(spuriousCollisions) >0:
                # print("spurious collision ahead")
                # print("Need to check the validity of the spurious collision, through the path")
            continue
            for spc in range(0, len(spuriousCollisions)):
                print("\n\n current spurious collision = ",
                      spuriousCollisions[spc])
                spcPath = spuriousCollisions[spc][0]
                spcObstacle = spuriousCollisions[spc][1]
                spcDnnOutput = spuriousCollisions[spc][2]
                # print("spcPath = ", spcPath)
                # print("spcObstacle = ", spcObstacle)
                # print("spcDnnOutput = ", spcDnnOutput)
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

                print("current region cons :", pd6.minimized_constraints())
                print("next region cons : ", pd7.minimized_constraints())

                pd7.poly_hull_assign(pd6)

                colStatus = collisionValidityCheck3_p3_1.checkValidityOfCollision(pathToCheckCollision, int(spcObstacle),
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
                    print("Details written to collisionData.txt file")
                    print("time Taken = ", datetime.now() - programStartTime)
                    sleep(10)
                    exit(0)
                elif(colStatus == 0):
                    print("True  >>>>>>>>>>>> Spurios Collision<<<<<<<<<<<<")
                    print("Program continue......")

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
        # print("region not processed")

    if intersectionRegionEqualsToRegion == 1:
        # print("intersectionRegionEqualsToRegion , ", intersectionRegionEqualsToRegion)
        # print("So continuing to next elment in the stack")
        continue
    
    # And(10*xp0>=1,100*xp0<=11,10*yp0>=45,100*yp0<=451, 10*zp0>=1945,100*zp0<=19451)

    tempFromula = And(True)
    for n in range(0, len(updatedExpString)):
        tempFromula = And(tempFromula, (updatedExpString[n]))

    # print("temp formula :", simplify(tempFromula))

    environment.groupCubeZ3[currGroup] = tempFromula
    # environment.groupCubeZ3[currGroupName+"_1"]  = tempFromula
    # environment.groupCubeZ3[currGroupName+"_2"]  = tempFromula

    # print("\n\n\nstarting new step :")
    # sleep(2)
    # main26.generateImagesInARegion(nextGroupName,tempFromula )
    # print("Pixel Intervals are generating with the following parmeters")
    # print(nextGroupName)
    # print(tempFromula)
    # sleep(10)
    # singleTriangleInvRegions30.computePixelIntervals(
    #     nextGroupName, tempFromula)

    # tempFromula =""
    intervalImageP3_4.computeIntervalImage_P3(nextGroupName, tempFromula,
                                           nextRegionMinMaxValues, nextRegionCornerPoints)



    # exit()
    # mt_singleTriangleInvRegions23.computePixelIntervals(nextGroupName, tempFromula)

#     absGroupNextPosImages6.findNextPosImages(currGroup,loopCount,pathLength)
    loopCount = loopCount+1
    # exit()
    # if loopCount == 1:
    #     print("loopCount = 1")
    #     sleep(2)
    #     exit()


for pre, fill, node in anytree.RenderTree(environment.A):
    print("%s%s" % (pre, node.name))



print("\n\nProgram Finished")
print("\nRegion is SAFE")
print(datetime.now())
print("time Taken = ", datetime.now() - programStartTime)




lFile = open("log.txt","a")
lFile.write("\n\nProgram Finished, Environment is SAFE \n")
lFile.write("\ntime Taken = "+str(datetime.now() - programStartTime))
lFile.write("\nNumber of spurious collisions = "+str(environment.spuriousCollisionCount))
lFile.write("\nNumber of regions interval image computed = "+str(whileLoopCount-targetReachCount-alreadyProcessedCount))
lFile.write("\nNumber of target reach = "+str(targetReachCount))
lFile.close()
















