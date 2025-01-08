
globalInervalImage = {}


def updateGlobalIntervalImage(numOfCurrInvRegions, fileName):
    print("Update global image")
    # singleTDataFile = open('singleTrianglePixelDatafromcpp.txt', 'r') 
    singleTDataFile = open(fileName, 'r') 
    
    line = singleTDataFile.readline()
    
    backGroundData = [1, 25, 24, 0, 1000]
    
    while line:
        
        # print(line)
        
        currentPixel = int(line)
        line = singleTDataFile.readline()
        numOfColours = int(line)
        
        currentPixelTrColors = []
        for i in range(0, numOfColours):
            
            #read each color details and create the interval for the current triangle.
            line = singleTDataFile.readline()
            colData = line.replace("\n","").split(",")
            currentPixelTrColors.append(colData)
            # print(colData)
        
        
        # print(currentPixelTrColors)
        #take union of these colours as interval
        
        if(numOfColours < numOfCurrInvRegions):
            currentPixelTrColors.append(backGroundData)
        
        # print(currentPixelTrColors) 
            
        currPixelInterval = [0]*8
        
        for k in range(0,3):
            
            kth_elements = [sublist[k] for sublist in currentPixelTrColors]
            
            kth_elements = [float(x) for x in kth_elements]
            
            currPixelInterval[2*k+0] = min(kth_elements)
            currPixelInterval[2*k+1] = max(kth_elements)
        
        minD_elements = [sublist[3] for sublist in currentPixelTrColors]
        minD_elements = [float(x) for x in minD_elements]
        currPixelInterval[6] = min(minD_elements)
         
        maxD_elements = [sublist[4] for sublist in currentPixelTrColors]
        max_elements = [float(x) for x in maxD_elements]
        currPixelInterval[7] = max(max_elements)
         
        # print("currPixelInterval ==>",currPixelInterval)        
        # print("\n\n\n")
        
        
        #read current interval data from the global interval images.
        
        if currentPixel in globalInervalImage:
            currentIntValue = globalInervalImage[currentPixel]
            
            #if current global depth's min is greater than the current pixel max depth then
            #assign the current triangle's interval as the global value.
            #if the current triangle's intervals min depth is greater than the global max depth
            # then do nothing           
            #else overlapping
            if currentIntValue[6] > currPixelInterval[7]:
                globalInervalImage[currentPixel] = currPixelInterval
                
            elif currPixelInterval[6] > currentIntValue[7] :
                pass
            else:
                tempList = []
                tempList.append(currentIntValue)
                tempList.append(currPixelInterval)
                
                tempIntervals = [0]*8
                
                for k in range(0,4):                
                    kth_elements = [sublist[2*k+0] for sublist in tempList]
                    k1th_elements = [sublist[2*k+1] for sublist in tempList]
                    
                    kth_elements = [float(x) for x in kth_elements]
                    k1th_elements = [float(x) for x in k1th_elements]
                    
                    tempIntervals[2*k+0] = min(kth_elements)
                    tempIntervals[2*k+1] = max(k1th_elements)
                
                globalInervalImage[currentPixel] = tempIntervals          
        else:
            globalInervalImage[currentPixel] = currPixelInterval
            
            
            
        
        line = singleTDataFile.readline()
        
        
            
          

numOfCurrRegions = 2
updateGlobalIntervalImage(numOfCurrRegions, 'singleTrianglePixelDatafromcpp.txt')

print("\n\n globalInervalImage ==>", globalInervalImage)
updateGlobalIntervalImage(numOfCurrRegions, "intervalDatatest1.txt")
print("\n\n globalInervalImage ==>", globalInervalImage)