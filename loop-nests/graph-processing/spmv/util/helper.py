# Helper functions

def getData(fname, statName, statType):
    "return sum of reads and updates for stat"
    
    try:
        f = open (fname, 'r')
    except FileNotFoundError:
        #print('File doesnt exist')
        return 0
    lines = f.readlines()
    f.close()

    numReads = 0
    numUpds  = 0

    searchStr = statName
    if statType == 'Tensor':
        searchStr += ':'
    else:
        searchStr += '_offchip:'
    
    events = []
    for l in range(len(lines)):
        if searchStr in lines[l]:
            lineCtr = l+1
            while (':\n' not in lines[lineCtr]):
                if lineCtr == len(lines):
                    break
                temp = lines[lineCtr].strip('\n')
                line = temp.split(' ')
                count = float(line[len(line)-1])
                events.append(count)
                lineCtr += 1
    assert(len(events) <= 2)
    return sum(events) 
