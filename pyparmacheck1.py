from pyparma import *



pd2 =NNC_Polyhedron(3,'empty')
xp0 = Variable(0)
yp0 = Variable(1)
zp0 = Variable(2)

pointList = []


tempList = [10,5,1]
pointList.append(tempList)

tempList = [-10,5,1]
pointList.append(tempList)

tempList = [10,-5,1]
pointList.append(tempList)

tempList = [-10,-5,1]
pointList.append(tempList)


tempList = [10,5,100]
pointList.append(tempList)


tempList = [-10,5,100]
pointList.append(tempList)


tempList = [10,-5,100]
pointList.append(tempList)

tempList = [-10,-5,100]
pointList.append(tempList)

for i in [0,1,3,2,4,5,7,6]:
        
        currPos = pointList[i]
        print(currPos)
        

        # print("currPos")
        mxp1 = currPos[0]
        myp1 = currPos[1]
        mzp1 = currPos[2]

        # mxp1 = str(mxp1).replace("?", "")
        # myp1 = str(myp1).replace("?", "")
        # mzp1 = str(mzp1).replace("?", "")

        ################

        mxp1 = int(float(mxp1)*pow(10, 14))
        myp1 = int(float(myp1)*pow(10, 14))
        mzp1 = int(float(mzp1)*pow(10, 14))

        # print(mxp1,myp1,mzp1)

        pd2.add_generator(point(mxp1*xp0+myp1 * yp0+mzp1*zp0, pow(10, 14)))
        
        print(pd2.minimized_constraints())
        
print(pd2.minimized_constraints())