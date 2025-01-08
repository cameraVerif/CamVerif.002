from pyparma import *



def getPolyhedraCornerPoints(polyConstraints):
    px = Variable(0)
    py = Variable(1)
    pz = Variable(2)


    pd = NNC_Polyhedron(3)

    for con in polyConstraints:
        pd.add_constraint(con)
   

    print(pd.minimized_constraints())

    print(pd.generators())

    cornerList = []
    for p in pd.generators():
        print(p)
        pstring = str(p).replace("closure_point(","").replace("point(","").replace(")","").replace(" ","")
        print(pstring)

        plist = pstring.split(",")
        print(plist)
        cornerList.append(plist)


    print(cornerList)

    finalCornerPoints = []
    for currPoint in cornerList:
        currCornerPoint = []
        for p in currPoint:
            a, b = p.split("/")
            print(a, b)  
            currCornerPoint.append(float(float(a)/float(b)))  
        finalCornerPoints.append(currCornerPoint)

    print(finalCornerPoints)
    return finalCornerPoints


# px = Variable(0)
# py = Variable(1)
# pz = Variable(2)


# pd2 = NNC_Polyhedron(3)
# pd2.add_constraint(px>=1)
# pd2.add_constraint(px<=2)

# pd2.add_constraint(py>=10)
# pd2.add_constraint(py<=40)

# pd2.add_constraint(pz>=100)
# pd2.add_constraint(pz<=400)

# allCons = pd2.minimized_constraints()

# print(allCons)
# getPolyhedraCornerPoints(allCons)



