import sys
import numpy as np
import matplotlib.pyplot as plt
import compiler

#stats    = ['parent', 'frontier', 'TileInOffsets', 'TileSources', 'Total']
stats    = ['parent', 'frontier']
versions = ['UNT-NOBUF', 'UNT-BUF', 'DST-T', 'SRC-T', 'DST-SRC-T']

fname = sys.argv[1]
f = open(fname, 'r')
lines = f.readlines()
f.close()

data = {} 
LINES_PER_GRAPH = len(versions) + 2

normalize = input('Normalize data set (wrt to unt with single buf) [Y/N]: ')

for l in range(len(lines)):
    if l % LINES_PER_GRAPH == 0:  
        temp = lines[l].strip('\n')
        line = temp.split(' ')
        graph = line[0]
        data[graph] = {}

        for s in range(len(stats)):
            stat = stats[s] 
            data[graph][stat] = {}
            for v in range(len(versions)):
                version = versions[v]
                data[graph][stat][version] = compiler.getStat(lines[l:], s, v)


graphs = []
for graph in data:
    graphs.append(graph)

if normalize == 'Y':
    for graph in data:
        for stat in data[graph]:
            for version in data[graph][stat]:
                for ind in range(len(data[graph][stat][version])):
                    try:
                        data[graph][stat][version][ind] = data[graph][stat]['UNT-BUF'][0] / data[graph][stat][version][ind]
                    except ZeroDivisionError:
                        data[graph][stat][version][ind] = 0



fig, ax = plt.subplots(len(stats), len(graphs), figsize=(30,10))
width = 0.4

for s in range(len(stats)):
    stat = stats[s]
    for g in range(len(graphs)):
        graph = graphs[g]

        total   = []
        offchip = []

        for version in versions:
            total.append(data[graph][stat][version][0])
            offchip.append(data[graph][stat][version][1])
        
        ind = np.arange(len(total))
        ax[s, g].bar(ind, total, width = width, label = 'total', color = 'b')
        ax[s, g].bar(ind+width, offchip, width = width, label = 'offchip', color = 'r')

        #ax[s, g].legend(loc = 'lower right')
        ax[s, g].set_xticks(np.arange(len(versions)) + width/2)
        if s == len(stats) - 1:
            ax[s, g].set_xticklabels(versions, rotation = 90)
        else:
            ax[s, g].set_xticklabels([])
            if s == 0:
                ax[s, g].set_title(graph)

        if g == 0:
            ax[s, g].set_ylabel(stat)
        
        
    
plt.tight_layout()

if normalize == 'Y':
    plt.savefig('../plots/plot-normalized.pdf')
else:
    plt.savefig('../plots/plot.pdf')
