
# import image_lb_file
# import image_ub_file
import environment

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


def generate_vnnlib_files3(finalGlobalIntervalImage):
    
    
    tempString = ""
    tempList = []
    dfImg = environment.defaultImg
    for i in range(0,49*49*3):
        # print(f"(declare-const X_{i} Real)")
        tempString += "(declare-const X_"+str(i)+" Real)\n"

    tempString += "(declare-const Y_0 Real)\n"
    tempString += "(declare-const Y_1 Real)\n"
    tempString += "(declare-const Y_2 Real)\n\n\n"
    
    for i in range(0,49*49):
        # print(f"(assert (<= X_{i} 0.679857769))")   
        # print(f"(assert (>= X_{i} 0.268978427))\n") 
        rD = dfImg[i][0]
        gD = dfImg[i][1]
        bD = dfImg[i][2]
        
        if finalGlobalIntervalImage.get(i):
            # print(i*3," ==> ", globalIntervalImage[i])
                    
            tempString += "(assert (>= X_"+str(i*3+0)+" "+str(min(rD/255,finalGlobalIntervalImage[i][0]/255))+"))\n"
            tempString += "(assert (<= X_"+str(i*3+0)+" "+str(max(rD/255,finalGlobalIntervalImage[i][1]/255))+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+1)+" "+str(min(gD/255,finalGlobalIntervalImage[i][2]/255))+"))\n"
            tempString += "(assert (<= X_"+str(i*3+1)+" "+str(max(gD/255,finalGlobalIntervalImage[i][3]/255))+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+2)+" "+str(min(bD/255,finalGlobalIntervalImage[i][4]/255))+"))\n"
            tempString += "(assert (<= X_"+str(i*3+2)+" "+str(max(bD/255,finalGlobalIntervalImage[i][5]/255))+"))\n"
            
            tempList.append(min(rD,finalGlobalIntervalImage[i][0]))
            tempList.append(max(rD,finalGlobalIntervalImage[i][1]))
            tempList.append(min(gD,finalGlobalIntervalImage[i][2]))
            tempList.append(max(gD,finalGlobalIntervalImage[i][3]))
            tempList.append(min(bD,finalGlobalIntervalImage[i][4]))
            tempList.append(max(bD,finalGlobalIntervalImage[i][5]))
            
        else:
            tempString += "(assert (>= X_"+str(i*3+0)+" "+str(min(rD/255,1/255))+"))\n"
            tempString += "(assert (<= X_"+str(i*3+0)+" "+str(max(rD/255,1/255))+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+1)+" "+str(min(gD/255,25/255))+"))\n"
            tempString += "(assert (<= X_"+str(i*3+1)+" "+str(max(gD/255,25/255))+"))\n"
            
            tempString += "(assert (>= X_"+str(i*3+2)+" "+str(min(bD/255,24/255))+"))\n"
            tempString += "(assert (<= X_"+str(i*3+2)+" "+str(max(bD/255,24/255))+"))\n"
            
            
            tempList.append(min(rD,1))
            tempList.append(max(rD,1))
            tempList.append(min(gD,25))
            tempList.append(max(gD,25))
            tempList.append(min(bD,24))
            tempList.append(max(bD,24))
        
        
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
    








# generate_vnnlib_files()