import sys
import os
import numpy as np
import helper

fname = sys.argv[1]

stats  = ['indegrees', 'vData', 'TileInOffsets', 'TileSources', 'Totals']
stats  = ['indegrees', 'vData']
stypes = ['Tensor', 'Offchip']

data = {}

# collect data
for stat in stats:
    data[stat] = {}
    for stype in stypes:
        data[stat][stype] = helper.getData(fname, stat, stype)

## print('***************** Stats **********************')
## print(stats)
## print('**********************************************')

for stat in stats:
    for stype in stypes:
        print(data[stat][stype], end=' ')

print('')

