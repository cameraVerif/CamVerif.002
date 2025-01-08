from pyparma import *

def getPoly():
    xp0 = Variable(0)
    yp0 = Variable(1)
    zp0 = Variable(2)
    pd3 = NNC_Polyhedron(3)
    pd3.add_constraint(1*10*xp0>=1*1)
    pd3.add_constraint(1*100*xp0<=11*1)
    pd3.add_constraint(1*2*yp0>=9*1)
    pd3.add_constraint(1*100*yp0<=451*1)
    pd3.add_constraint(1*2*zp0>=383*1)
    pd3.add_constraint(1*100*zp0<=19151*1)
    return pd3
