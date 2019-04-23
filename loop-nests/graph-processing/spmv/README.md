# Sparse Matrix Multiplying Dense Vector 

This is a proxy for the pagerank-classic implementation (all vertices are processed every iteration). 
Note that the result of the computation is just the in-degrees of the nodes in the graph.

The app works on CSC version of directed graphs (uses `<graphname>.in-offsets` and `<graphname>.sources` as inputs)

**NOTE:** Since adjmatrix is not suited for large graphs, this directory only contains compressed 
representations of graphs 
