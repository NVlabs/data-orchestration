# data consolidation functions

import numpy as np

def getStat(lines, statInd, verInd):
    "return data for graph, stat, and version"
    
    temp = lines[1+verInd].strip('\n')
    line = temp.split(' ')

    retVals = np.zeros(2)

    retVals[0] = int(line[(2*statInd)])
    retVals[1] = int(line[(2*statInd)+1])

    return retVals
    
