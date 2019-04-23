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
    
    for l in range(len(lines)):
        if statName in lines[l]:
            lineCtr = l
            readCtr = 0
            updCtr  = 0
            while (True):
                lineCtr += 1
                if lineCtr == len(lines):
                    break
                if statType in lines[lineCtr]:
                    if 'reads' in lines[lineCtr]:
                        readCtr += 1
                        temp = lines[lineCtr].strip('\n')
                        line = temp.split(' ')
                        numReads = int(line[len(line)-1])
                    elif 'updates' in lines[lineCtr]:
                        updCtr += 1
                        temp = lines[lineCtr].strip('\n')
                        line = temp.split(' ')
                        numUpds = int(line[len(line)-1])
                if readCtr == 1 and updCtr == 1:
                    break

            break

    return numReads + numUpds

                
            
