from pyparma import *

def getPoly():
    xp0 = Variable(0)
    yp0 = Variable(1)
    zp0 = Variable(2)
    pd3 = NNC_Polyhedron(3)
    pd3.add_constraint(10* xp0 == 1)
    pd3.add_constraint(2* yp0 == 9)
    pd3.add_constraint(2* zp0 == 243)
    return pd3
