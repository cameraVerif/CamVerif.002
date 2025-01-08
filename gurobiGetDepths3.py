#!/usr/bin/env python3.7


import gurobipy as gp
from gurobipy import GRB
import math

from time import sleep
from decimal import Decimal

# Create a new model
# m = gp.Model("qcp")



def getDepthInterval(currImageSetConString, vert_x, vert_y, vert_z):
    m = gp.Model("model1")

    # m.setParam(GRB.Param.NonConvex, 2)
    m.setParam(GRB.Param.NumericFocus, 3)
    # # # m.setParam(GRB.Param.OutputFlag, 0)
    # m.setParam('BarHomogeneous',1)
    # m.setParam('Method',5)

    # Create variables
    x = m.addVar(lb=-GRB.INFINITY, name="x")
    y = m.addVar(lb=-GRB.INFINITY, name="y")
    z = m.addVar(lb=-GRB.INFINITY, name="z")

    xp0 = m.addVar(lb=-GRB.INFINITY, name="xp0")
    yp0 = m.addVar(lb=-GRB.INFINITY, name="yp0")
    zp0 = m.addVar(lb=-GRB.INFINITY, name="zp0")

    minVal =0
    maxVal =0
    # Set objective: x
    obj = ((xp0 - x)**2 + (yp0 - y)**2 + (zp0 - z)**2)

    m.addConstr(x == vert_x, "c0")
    m.addConstr(y == vert_y, "c1")
    m.addConstr(z == vert_z, "c2")
    
    print(vert_x, vert_y, vert_z)

    consList = currImageSetConString.split(",")

    for i in range(0, len(consList)):
        consName = "q_"+str(i)
        print(consName, " :", consList[i])
        currCons = consList[i]
        try:
            exec(f"m.addConstr({currCons})")
        except NotImplementedError:
            currCons = currCons.replace("<","+0.1<=")
            currCons = currCons.replace(">","-0.1>=")
            # currCons = currCons.replace("<","<=")
            # currCons = currCons.replace(">",">=")
            exec(f"m.addConstr({currCons})")
            print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(20)
            exit(0)
        # m.addConstr(currCons, "str(consName)")

    print("added")
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    minVal = obj.getValue()

    # for v in m.getVars():
    #     print('%s %g' % (v.VarName, v.X))

    # print('Obj: %g' % obj.getValue())
    
    print("\n\nminimum value got ", minVal)

    # m.setObjective(obj, GRB.MAXIMIZE)
    # m.setParam('TimeLimit', 0.1*60)
    # m.setParam('BarHomogeneous',1)
    
    
    # m.optimize()
    # print(m.Status)
    # if (m.Status == 9):
    #     print("Gurobi timeout happened")
    #     maxVal = obj.getValue()
    #     print("maxVal = ",maxVal)
    #     sleep(2)

    # # for v in m.getVars():
    # #     print('%s %g' % (v.VarName, v.X))

    # # print('Obj: %g' % obj.getValue())
    # maxVal = obj.getValue()

    return minVal

def getDepthInterval2(currImageSetConString, vert_x, vert_y, vert_z):
    m = gp.Model("model1")

    # m.setParam(GRB.Param.NonConvex, 2)
    # m.setParam(GRB.Param.NumericFocus, 3)
    # # m.setParam(GRB.Param.OutputFlag, 0)
    # m.setParam('BarHomogeneous',1)
    # m.setParam('Method',5)

    # Create variables
    x = m.addVar(lb=-GRB.INFINITY, name="x")
    y = m.addVar(lb=-GRB.INFINITY, name="y")
    z = m.addVar(lb=-GRB.INFINITY, name="z")

    xp0 = m.addVar(lb=-GRB.INFINITY, name="xp0")
    yp0 = m.addVar(lb=-GRB.INFINITY, name="yp0")
    zp0 = m.addVar(lb=-GRB.INFINITY, name="zp0")

    minVal =0
    maxVal =0
    # Set objective: x
    obj = ((xp0 - x)**2 + (yp0 - y)**2 + (zp0 - z)**2)

    m.addConstr(x == -vert_x, "c0")
    m.addConstr(y == -vert_y, "c1")
    m.addConstr(z == -vert_z, "c2")

    consList = currImageSetConString.split(",")

    for i in range(0, len(consList)):
        consName = "q_"+str(i)
        print(consName, " :", consList[i])
        currCons = consList[i]
        try:
            exec(f"m.addConstr({currCons})")
        except NotImplementedError:
            # currCons = currCons.replace("<","+0.00000000000000000000000001<=")
            # currCons = currCons.replace(">","-0.00000000000000000000000001>=")
            currCons = currCons.replace("<","<=")
            currCons = currCons.replace(">",">=")
            # currCons = currCons.replace("<","+0.1<=")
            # currCons = currCons.replace(">","-0.1>=")
            # exec(f"m.addConstr({currCons})")
            print("Exception handled")
            # return 0,0
            # sleep(2)
        except OverflowError:
            
            print("overflow error")
            
            sleep(2)
            exit(0)
        # m.addConstr(currCons, "str(consName)")

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    minVal = obj.getValue()

    # for v in m.getVars():
    #     print('%s %g' % (v.VarName, v.X))

    # print('Obj: %g' % obj.getValue())
    
    print("\n\nminimum value got ", minVal)

    # m.setObjective(obj, GRB.MAXIMIZE)
    # m.setParam('TimeLimit', 0.1*60)
    # m.setParam('BarHomogeneous',1)
    
    
    # m.optimize()
    # print(m.Status)
    # if (m.Status == 9):
    #     print("Gurobi timeout happened")
    #     maxVal = obj.getValue()
    #     print("maxVal = ",maxVal)
    #     sleep(2)

    # # for v in m.getVars():
    # #     print('%s %g' % (v.VarName, v.X))

    # # print('Obj: %g' % obj.getValue())
    # maxVal = obj.getValue()

    return minVal



# tempString = "-5*yp0+23>=0, \
#             -5*zp0+973>=0, \
#             -67*xp0+20*zp0-3882>=0,\
#                 2*zp0-389>=0,\
#             2*yp0-9>=0, \
#             10*xp0-1>=0, \
#             446032970269918841442500154105960108452677559398580342531204223632812500000*xp0-111508242567479609337750771548209982597654743585735559463500976562500000000*yp0-156444400020046202685119627218842779103891160730199771933257579803466796875*zp0+30901097608214865008506245858035186465442304049579719035827940549281760411648>=0"


# minVal, maxVal = getDepthInterval(tempString, 1, 6, 190)

# print(minVal, maxVal)


# m.addConstr(18640183871009435028879002379633607576481235668097724555991590023040771484375*xp0-21303067227887086932031557266479455045593560669203725410625338554382324218750*yp0-13910584928943421230912293046211303249037882778793573379516601562500*zp0+95863802709139921281855567370715357663903522101174430064229129852916978417664>=0, "q6")


# # Add rotated cone: x^2 <= yz
# m.addConstr(x**2 <= y*z, "qc1")




currImageSetConString = "-34355348946069201581662874189307533468172968991647144418324310925608124000317962365794269271549179150221827113987885785071967689835991923901582135658550267442328108326410701161853321482609516496874907885971066314073339022299203362193318068499304094852052026152729717348016891*xp0+325531005937665099584915846115143628494462361395489559826202236237821479664749744784673524272183506993127591132138110113311446342151821369886448520524101003931991858539327359030336645273974194513364392400496656592035505485253404457673965080104601725759015109231785904136115930972717713865438309013*yp0-111749449799496974484382300690021500934297625643034161142360331357817319202984139089885086391704861914026676184291362977202489777661428323581335335150622372555496090456344269581743583448278045448148816094852908266877113801298564769963080153235263088750036426886971114338034317086432105886308199665*zp0+20236367757169778205541530546895650751021045745545364012968503252507540195117586820686959196173512039074878209585287325229397338228892953826434469050259923522050993531834993894812809693534165400049503329016886293062844058677340426414475003132250704214334267392176397046346754569009247958367314480020>=0, -67*xp0-16*zp0+3145>=0, -67*yp0-8*zp0+1874>0, 2*zp0-389>=0, 1124013462316997193384902512371404537650055738665740027331310893021376450813573696923437265693189073740872021329430990062132288935580348412887894861362239064035126772070687005343789587781472490753230297334930850978292010778370368353298220469697138446307327617738085930047943837910101772995121*xp0+2231861750688489731855714195283693317259890937037177135723113332798599375001279135439330039001348604359407159944334764061103462622286546191547316982274691337608199810831655907871187997100113355555965909526423051740968160792205171472446084680335633074092543905768712288525523632899607179629462321487905460599796097*yp0-766161496505003937801265649730028317739666362080480422914097863008713176792707672354979925766917292856076641649504166278008988901142967990683204290434696146421302378115107053704916855427146555803033541610016038918127274464369010991331266025075442862823707529762529463454118888649913125212951898891575461180810382*zp0+138741853606232234823577165355041517146838719685771220680470893460804625154856932577876976302111433281158694903564431831092550495188019564094162272457709399603911437762348041973833513401494852912202205746716528109194450340651692850932506863933545853685469641570536708987032351878684407859553687074561979396669314191>=0, 5231589143591765317442499622901165443533230787899883682988237825007475239892319228011718873857343392693140776749306090791472403343343446234314467272307691892778198818131937310490703171799794956730169746350162978974026879215882362186236083606118672029761894498352849373165787279639644422400000*xp0-234250260160825312721305953264231288516413318861188821626339007089886951039954592299032188381672092210140631794745048841409212090000452816461841818163030980273650693349191222857792679334319177167022525955977446819732546830561896814309078370423224120735607216344157434619363609536103481600000*zp0+43102047869591857540720295400618557087020050670458743179246377304539198991351644983021922662227664966665876250233088986819295024560083318228978894541997700370351727576251185005833852997514728598732144775899850214830788616823389013832870420157873238215351727807324967969962904154643040614400000>0 "

getDepthInterval(currImageSetConString, 2,6,184)



