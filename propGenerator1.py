




tempString = ""

for i in range(0,49*49*3):
    # print(f"(declare-const X_{i} Real)")
    tempString += "(declare-const X_"+str(i)+" Real)\n"
    
tempString += "(declare-const Y_0 Real)\n"
tempString += "(declare-const Y_1 Real)\n"
tempString += "(declare-const Y_2 Real)\n\n\n"
    
for i in range(0,49*49):
    tempString += "(assert (>= X_"+str(i*3+0)+" "+str(100/255)+"))\n"
    tempString += "(assert (<= X_"+str(i*3+0)+" "+str(100/255)+"))\n"
    
    tempString += "(assert (>= X_"+str(i*3+1)+" "+str(150/255)+"))\n"
    tempString += "(assert (<= X_"+str(i*3+1)+" "+str(150/255)+"))\n"
    
    tempString += "(assert (>= X_"+str(i*3+2)+" "+str(20/255)+"))\n"
    tempString += "(assert (<= X_"+str(i*3+2)+" "+str(20/255)+"))\n"
    
    
    

f = open("prop_y2_2.vnnlb", "w")
f.write(tempString)
tempString2 = "(assert (or\n"
tempString2 += " (and (>= Y_2 Y_0) (>= Y_2 Y_1))))"
f.write(tempString2)
f.close()       














