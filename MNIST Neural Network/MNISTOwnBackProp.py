import sys; args = sys.argv[1:]
import random
import math
import csv

def setGlobals():
    global inputs
    inputs = []
    global outputs
    outputs = []
    
    csvlist = []
    with open("mnist_test.csv", mode = "r") as file:
        csvlist = csv.reader(file)
        for i, line in enumerate(csvlist):
            if i == 0: continue
            inputs.append([int(i)/255 for i in line[1:]])           #divide each grayscale value by 255 to scale it down so it fits between (0, 1)
            outputs.append(int(line[0]))
    
    global layerCt, weightCt
    layerCt = [len(inputs[0][1])+1, 2, 1, 1]
    weightCt = [2*(len(inputs[0][1])+1), 2, 1]
    
    global weights
    weights = [[random.random() for i in range(x)] for x in weightCt]

def transfer(putty):
    log = 1/(1 + math.exp(-putty))
    return log
       
def dotProduct(v1, v2):
    return sum(v1[i]*v2[i] for i in range(len(v1)))
       
def feedForward(inputty):           #Need to make more than one output cases accepectable
    curList = inputty
    outputty = []
    updatedNums = []
    for idx, lcount in enumerate(layerCt[1:]):  # 2 1 1
        newList = []
        for no in range(lcount):  #for each iteration (get weights)
            sum = dotProduct(curList, weights[idx][no*len(curList):no*len(curList)+len(curList)])
            if idx < len(layerCt[1:])-1:
                newList.append(transfer(sum))
            else:
                outputty = sum
                newList.append(sum)
        updatedNums.append(curList)
        curList = newList       #switching to another layer
    return (updatedNums, outputty)       #returns (filled out Neural Networks, actual value found)

def backProp(expected, outputty, network):
    error = [n - outputty[i] for i, n in enumerate(expected)]      #expected - actual --> error = [for i, n in enumerate(expected)]
    newWeights = [[*weigh] for weigh in weights]
    
    step3 = error       #partial 2, partial wrt the y
    errorCells = [[], [], [], [], [], []]
    for i, nodes in enumerate(network[::-1]):       #reversing the neural network to back propagate
        errorCell = 0
        for ind, node in enumerate(nodes):
            #print("err", errorCells)
            if i == 0:
                errorCell = step3
                errorCells[i].append(errorCell)
            if i == 1 and ind==0:
                errorCell = step3*weights[len(weights)-i][0]*network[len(network)-i][0]*(1-network[len(network)-i][0])
                if ind == 0: errorCells[i].append(errorCell)
            if i >= 2:
                for indy,n in enumerate(network[len(network)-i]):
                    #print(weights[len(weights)-i][indy::len(network[i-1])])
                    err = dotProduct(errorCells[i-1], weights[len(weights)-i][indy::len(network[len(network)-i])])      #dot product part
                    errorCell = err * n * (1 - n)
                    if ind==0: errorCells[i].append(errorCell)
                    
                    partial = errorCell*node
                    for ii,w in enumerate(weights[len(weights)-i-1][indy*len(nodes):indy*len(nodes)+len(nodes)]):
                        newWeights[len(weights)-i-1][indy*len(nodes)+ii] = w + 0.1*partial  #FIX THIS, MAKE GENERAL
                continue
                    
            partial = errorCell*node
            constant = weightCt[len(network)-i-1]//len(nodes)
            # print("errorcell", errorCell)
            # print("node", node)
            # print("partial", partial)
            for ii, w in enumerate(weights[len(weights)-i-1][ind*constant:ind*constant+constant]):       
                newWeights[len(weights)-i-1][ind*constant+ii] = w + 0.1*partial           #updating weight
                #print("new Weight", w+0.1*partial)
            #do partial part
        step3 = errorCell
        #print(newWeights)
    #print("errorCells", errorCells)
    return newWeights    

def main():
    global weights
    if len(inputs[0]) < layerCt[0]:         #adding bias to the inputs (1)
        for idx, inputList in enumerate(inputs):
            inputList.append(1)
    
    #print("inputs: ", inputs)
    
    print("Layer counts: " + str(layerCt[0]) + " 2 1 1")
    
    for i in range(1):
        accuracyCt = 0
        for index, inputty in enumerate(inputs):
            expectedValue = [0]*10
            expectedValue[outputs[index]] = 1
            updatedNums, actualValue = feedForward(inputty)
            #print(updatedNums, actualValue)
            #print(actualValue)
            #print(updatedNums)
            #print(inputty)
            #print(weights)
            newWeights = backProp(expectedValue, actualValue, updatedNums)
            weights = [[w for w in we] for we in newWeights]
            print("updated weights", weights)
    
    
    finalStr = ""
    for weightList in weights:
        for w in weightList:
            finalStr = finalStr + str(w) + " "
        finalStr+="\n"
    
    print(finalStr)
       
    
setGlobals()
if __name__ == "__main__": main()
# Yuan Yin Student, Period 3, 2025