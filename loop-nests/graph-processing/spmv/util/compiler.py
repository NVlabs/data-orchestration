# data consolidation functions

import numpy as np

def getStat(lines, statInd, verInd):
    "return data for graph, stat, and version"
    
    temp = lines[1+verInd].strip('\n')
    line = temp.split(' ')

    retVals = np.zeros(2, dtype=int)

    retVals[0] = float(line[(2*statInd)])
    retVals[1] = float(line[(2*statInd)+1])

    return retVals
    
