
# import image_lb_file
# import image_ub_file

from pyparma import *
import environment
import pythonRenderAnImage2



# from importlib import reload  # Python 3.4+

def generate_vnnlib_files(globalIntervalImage):
    # global image_lb_file
    # global image_ub_file
    # image_lb_file = reload(image_lb_file)
    # image_ub_file = reload(image_ub_file)
    
    # print("inside vnn lb generate fuc=nction")
    # print("globalInervalImage =", globalIntervalImage)
    #
    #
    
    tempString = ""
    
    tempList = []

    for i in range(0,49*49*3):
        # print(f"(declare-const X_{i} Real)")
        tempString += "(declare-const X_"+str(i)+" Real)\n"

    f0 = open("globalMin.txt", "w")
    for i in range(0,49*49):
        
        if i in globalIntervalImage:
            
            if(i==1451):
                print("1451 presentin global interval iamge, writing min of 1452")
                print(str(globalIntervalImage[i][0]))
                print(str(globalIntervalImage[i][2]))
                print(str(globalIntervalImage[i][4]))
            f0.write(str(int(globalIntervalImage[i][0]))+"\n")
            f0.write(str(int(globalIntervalImage[i][2]))+"\n")
            f0.write(str(int(globalIntervalImage[i][4]))+"\n")
        else:
            f0.write(str("1\n25\n24\n"))
            
    f0.close()  
    
    f0 = open("globalMax.txt", "w")
    for i in range(0,49*49):
        if i in globalIntervalImage:
            f0.write(str(int(globalIntervalImage[i][1]))+"\n")
            f0.write(str(int(globalIntervalImage[i][3]))+"\n")
            f0.write(str(int(globalIntervalImage[i][5]))+"\n")
        else:
            f0.write(str("1\n25\n24\n"))
            
    f0.close()  
        

    # print("(declare-const Y_0 Real)")
    # print("(declare-const Y_1 Real)")
    # print("(declare-const Y_2 Real)")

    tempString += "(declare-const Y_0 Real)\n"
    tempString += "(declare-const Y_1 Real)\n"
    tempString += "(declare-const Y_2 Real)\n\n\n"
    
    for i in range(0,49*49):
        # print(f"(assert (<= X_{i} 0.679857769))")   
        # print(f"(assert (>= X_{i} 0.268978427))\n") 
        
        if i in globalIntervalImage:
            # print(i*3," ==> ", globalIntervalImage[i])
                    
            tempString += "(assert (>= X_"+str(i*3+0)+" "+str(globalIntervalImage[i][4]/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+0)+" "+str(globalIntervalImage[i][5]/255)+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+1)+" "+str(globalIntervalImage[i][2]/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+1)+" "+str(globalIntervalImage[i][3]/255)+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+2)+" "+str(globalIntervalImage[i][0]/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+2)+" "+str(globalIntervalImage[i][1]/255)+"))\n"
            
            tempList.append(globalIntervalImage[i][4]/255)
            tempList.append(globalIntervalImage[i][5]/255)
            tempList.append(globalIntervalImage[i][2]/255)
            tempList.append(globalIntervalImage[i][3]/255)
            tempList.append(globalIntervalImage[i][0]/255)
            tempList.append(globalIntervalImage[i][1]/255)
            

        else:
            tempString += "(assert (>= X_"+str(i*3+0)+" "+str(24/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+0)+" "+str(24/255)+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+1)+" "+str(25/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+1)+" "+str(25/255)+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+2)+" "+str(1/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+2)+" "+str(1/255)+"))\n"
            
            tempList.append(24/255)
            tempList.append(24/255)
            tempList.append(25/255)
            tempList.append(25/255)
            tempList.append(1/255)
            tempList.append(1/255)
        
        
    fff = open("mrb2.vnnlib","w")
    for i in range(0,49*49):
        fff.write("x"+str(i*3+0)+" >= "+str(tempList[i*6+0]))
        fff.write("x"+str(i*3+0)+" <= "+str(tempList[i*6+1]))
        
        fff.write("x"+str(i*3+1)+" >= "+str(tempList[i*6+2]))
        fff.write("x"+str(i*3+1)+" <= "+str(tempList[i*6+3]))
        
        fff.write("x"+str(i*3+2)+" >= "+str(tempList[i*6+4]))
        fff.write("x"+str(i*3+2)+" <= "+str(tempList[i*6+5]))
        
        
    fff.close()
#         x0 >= 0.6
# x0 <= 0.6798577687
              
        
        
    
    f = open("prop_y0.vnnlb", "w")
    f.write(tempString)

    tempString2 = "(assert (or\n"
    tempString2 += " (and (>= Y_0 Y_1) (>= Y_0 Y_2))))"
    f.write(tempString2)
    f.close()   

    # print("(assert (or")
    # print(" (and (>= Y_0 Y_1) (>= Y_0 Y_2))))")   
        
    f = open("prop_y1.vnnlb", "w")
    f.write(tempString)
    tempString2 = "(assert (or\n"
    tempString2 += " (and (>= Y_1 Y_0) (>= Y_1 Y_2))))"
    f.write(tempString2)
    f.close()  

    f = open("prop_y2.vnnlb", "w")
    f.write(tempString)
    tempString2 = "(assert (or\n"
    tempString2 += " (and (>= Y_2 Y_0) (>= Y_2 Y_1))))"
    f.write(tempString2)
    f.close()  

    del tempString2
    del tempString
    


def generate_vnnlib_files2(finalGlobalIntervalImage):
    
    
    tempString = ""
    tempList = []

    for i in range(0,49*49*3):
        # print(f"(declare-const X_{i} Real)")
        tempString += "(declare-const X_"+str(i)+" Real)\n"

    tempString += "(declare-const Y_0 Real)\n"
    tempString += "(declare-const Y_1 Real)\n"
    tempString += "(declare-const Y_2 Real)\n\n\n"
    
    for i in range(0,49*49):
        # print(f"(assert (<= X_{i} 0.679857769))")   
        # print(f"(assert (>= X_{i} 0.268978427))\n") 
        
        if finalGlobalIntervalImage.get(i):
            # print(i*3," ==> ", globalIntervalImage[i])
                    
            tempString += "(assert (>= X_"+str(i*3+0)+" "+str(finalGlobalIntervalImage[i][0]/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+0)+" "+str(finalGlobalIntervalImage[i][1]/255)+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+1)+" "+str(finalGlobalIntervalImage[i][2]/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+1)+" "+str(finalGlobalIntervalImage[i][3]/255)+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+2)+" "+str(finalGlobalIntervalImage[i][4]/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+2)+" "+str(finalGlobalIntervalImage[i][5]/255)+"))\n"
            
            tempList.append(finalGlobalIntervalImage[i][0]/255)
            tempList.append(finalGlobalIntervalImage[i][1]/255)
            tempList.append(finalGlobalIntervalImage[i][2]/255)
            tempList.append(finalGlobalIntervalImage[i][3]/255)
            tempList.append(finalGlobalIntervalImage[i][4]/255)
            tempList.append(finalGlobalIntervalImage[i][5]/255)
            
        else:
            tempString += "(assert (>= X_"+str(i*3+0)+" "+str(1/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+0)+" "+str(1/255)+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+1)+" "+str(25/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+1)+" "+str(25/255)+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+2)+" "+str(24/255)+"))\n"
            tempString += "(assert (<= X_"+str(i*3+2)+" "+str(24/255)+"))\n"
            
            
            tempList.append(1/255)
            tempList.append(1/255)
            tempList.append(25/255)
            tempList.append(25/255)
            tempList.append(24/255)
            tempList.append(24/255)
        
        
    fff = open("mrb2.vnnlib","w")
    for i in range(0,49*49):
        fff.write("x"+str(i*3+0)+" >= "+str(tempList[i*6+0])+"\n")
        fff.write("x"+str(i*3+0)+" <= "+str(tempList[i*6+1])+"\n")
        
        fff.write("x"+str(i*3+1)+" >= "+str(tempList[i*6+2])+"\n")
        fff.write("x"+str(i*3+1)+" <= "+str(tempList[i*6+3])+"\n")
        
        fff.write("x"+str(i*3+2)+" >= "+str(tempList[i*6+4])+"\n")
        fff.write("x"+str(i*3+2)+" <= "+str(tempList[i*6+5])+"\n")
        
        
    fff.close()
#       
        
        
    
    f = open("prop_y0.vnnlb", "w")
    f.write(tempString)

    tempString2 = "(assert (or\n"
    tempString2 += " (and (>= Y_0 Y_1) (>= Y_0 Y_2))))"
    f.write(tempString2)
    f.close()   

    # print("(assert (or")
    # print(" (and (>= Y_0 Y_1) (>= Y_0 Y_2))))")   
        
    f = open("prop_y1.vnnlb", "w")
    f.write(tempString)
    tempString2 = "(assert (or\n"
    tempString2 += " (and (>= Y_1 Y_0) (>= Y_1 Y_2))))"
    f.write(tempString2)
    f.close()  

    f = open("prop_y2.vnnlb", "w")
    f.write(tempString)
    tempString2 = "(assert (or\n"
    tempString2 += " (and (>= Y_2 Y_0) (>= Y_2 Y_1))))"
    f.write(tempString2)
    f.close()  

    del tempString2
    del tempString

def ensureBackground():
    region = environment.initRegionPolyhedron
    gs = region.minimized_generators()
    vertString = str(gs)
    vertString = vertString.replace("Generator_System {","").replace("}","").replace("point","").replace("(","").replace(")","")
    points = vertString.split(",")
    points = [item.replace("closure_","") for item in points] 
    points = [item.strip() for item in points]
    points = [eval(item) for item in points]
    defIvImg = dict()
    for i in range(0, 49*49):
        defIvImg[i] = []
    for i in range(0, int(len(points)/3)):
        xpp = points[i*3+0]
        ypp = points[i*3+1]
        zpp = points[i*3+2]
        pythonRenderAnImage2.renderAnImage(xpp,ypp,zpp,"xpptestImage1")
        for key in defIvImg:
            defIvImg[key].append(environment.defaultImg[key])
    
    for key, value in defIvImg.items():
        transposed = list(zip(*value))
        min_list = [min(group) for group in transposed]
        max_list = [max(group) for group in transposed]
        defIvImg[key] = [min_list, max_list]  
    environment.defIVImag = defIvImg

def generate_vnnlib_files3(finalGlobalIntervalImage):
    
    
    tempString = ""
    tempList = []
   
    for i in range(0,49*49*3):
        # print(f"(declare-const X_{i} Real)")
        tempString += "(declare-const X_"+str(i)+" Real)\n"

    tempString += "(declare-const Y_0 Real)\n"
    tempString += "(declare-const Y_1 Real)\n"
    tempString += "(declare-const Y_2 Real)\n\n\n"
    ensureBackground()
    dfImg = environment.defIVImag
    
    for i in range(0,49*49):
        # print(f"(assert (<= X_{i} 0.679857769))")   
        # print(f"(assert (>= X_{i} 0.268978427))\n") 
        currentValue = dfImg[i]
        rD1 = currentValue[0][0]
        rD2 = currentValue[1][0]
        gD1 = currentValue[0][1]
        gD2 = currentValue[1][1]
        bD1 = currentValue[0][2]
        bD2 = currentValue[1][2]
        
        if finalGlobalIntervalImage.get(i):
            # print(i*3," ==> ", globalIntervalImage[i])
                    
           
            
            tempList.append(min(rD1,finalGlobalIntervalImage[i][0]))
            tempList.append(max(rD2,finalGlobalIntervalImage[i][1]))
            tempList.append(min(gD1,finalGlobalIntervalImage[i][2]))
            tempList.append(max(gD2,finalGlobalIntervalImage[i][3]))
            tempList.append(min(bD1,finalGlobalIntervalImage[i][4]))
            tempList.append(max(bD2,finalGlobalIntervalImage[i][5]))
            
        else:
           
            
            tempList.append(min(rD1,1))
            tempList.append(max(rD2,1))
            tempList.append(min(gD1,25))
            tempList.append(max(gD2,25))
            tempList.append(min(bD1,24))
            tempList.append(max(bD2,24))
        
        
    # fff = open("mrb2.vnnlib","w")
    # for i in range(0,49*49):
    #     fff.write("x"+str(i*3+0)+" >= "+str(tempList[i*6+0])+"\n")
    #     fff.write("x"+str(i*3+0)+" <= "+str(tempList[i*6+1])+"\n")
        
    #     fff.write("x"+str(i*3+1)+" >= "+str(tempList[i*6+2])+"\n")
    #     fff.write("x"+str(i*3+1)+" <= "+str(tempList[i*6+3])+"\n")
        
    #     fff.write("x"+str(i*3+2)+" >= "+str(tempList[i*6+4])+"\n")
    #     fff.write("x"+str(i*3+2)+" <= "+str(tempList[i*6+5])+"\n")
        
        
    # fff.close()
#       

    f0 = open("globalMin.txt", "w")
    f1 = open("globalMax.txt", "w")
    for i in range(0,49*49):
        f0.write(str(int(tempList[i*6+0]))+"\n")
        f1.write(str(int(tempList[i*6+1]))+"\n")
        f0.write(str(int(tempList[i*6+2]))+"\n")
        f1.write(str(int(tempList[i*6+3]))+"\n")
        f0.write(str(int(tempList[i*6+4]))+"\n")
        f1.write(str(int(tempList[i*6+5]))+"\n")
        
        # if i in finalGlobalIntervalImage:
          
        #     f0.write(str(int(finalGlobalIntervalImage[i][0]))+"\n")
        #     f0.write(str(int(finalGlobalIntervalImage[i][2]))+"\n")
        #     f0.write(str(int(finalGlobalIntervalImage[i][4]))+"\n")
        # else:
        #     f0.write(str("1\n25\n24\n"))
            
    f0.close() 
    f1.close() 
     
    
    # f0 = open("globalMax.txt", "w")
    # for i in range(0,49*49):
    #     if i in finalGlobalIntervalImage:
    #         f0.write(str(int(finalGlobalIntervalImage[i][1]))+"\n")
    #         f0.write(str(int(finalGlobalIntervalImage[i][3]))+"\n")
    #         f0.write(str(int(finalGlobalIntervalImage[i][5]))+"\n")
    #     else:
    #         f0.write(str("1\n25\n24\n"))
            
    # f0.close()  
        
        
    








# generate_vnnlib_files()