################# -- De-duplicated pipelined indexed version -- ##############

# Parameters:
V = 1M  # number of vertices
F = 32  # feature size per vertex
H = 64  # hidden layer size
B = 256 # batch size
N = 12  # neighbor samples per vertex

# Inputs:
float features[V][F]
bool  is_connected[V, V]

# Intermediate tensors:

# -- dense --
int   batch[B]
int   n1_sample[B][N]
int   n2_sample[B][N][N]

# -- sparse --
bool  n1_dedup[V]
bool  n2_dedup[V]

# -- dense --
float n1_encoded[B][N][H]
float n2_encoded[B][N][N][H]
float n1_sums[B][N][H]
float src_sums[B][H]

# Output:
float prediction[B]

# Pre-generate (in un-simulated code) batch, n1 and n2 samples.
for b_pos = [0..B):
  s = rand() % V
  batch[b_pos] = s
  
  for n1_pos = [0..N):
    n1_idx = rand() % adj_count[s]
    n1 = adj_lookup[s, n1_idx]
    n1_sample[b_pos, n1_pos] = n1
    
    for n2_pos = [0..N):
      n2_idx = rand() % adj_count[n1]
      n2 = adj_lookup[n1, n2_idx]
      n2_sample[b_pos, n1_pos, n2_pos] = n2
      
# -- sparse --
            
# Dedup n1 neighbors.
for b_pos = [0..B):
  for n1_pos = [0..N):
    n1 = n1_sample[b_pos, n1_pos]
    n1_dedup[n1] = true

# Dedup n2 neighbors.
for b_pos = [0..B):
  for n1_pos = [0..N):
    for n2_pos = [0..N):
      n2 = n2_sample[b_pos, n1_pos, n2_pos]
      n2_dedup[n2] = true

# Fetch and encode de-duplicated n2 vertices.
for n2 = [0..V]:
  if n2_dedup[n2]:    
    for h = [0..H):
      for f = [0..F):
        temp[h] += W_in[h][f] * features[n2][f]
      n2_encoded[n2][h] = ReLU(temp[h])

# Fetch and encode de-duplicated n1 vertices.
for n1 = [0..V]:
  if n1_dedup[n1]:
    for h = [0..H):
      for f = [0..F):
        temp[h] += W_in[h][f] * features[n1][f];
      n1_encoded[n1][h] = ReLU(temp[h])

# -- dense --

# (1) Work on B*N*N cuboid.
for b_pos = [0..B):
  for n1_pos = [0..N):
    for n2_pos = [0..N):
      
      # Gather n2_encoded and reduce into n1_sums.
      n2_idx = n2_sample[b_pos, n1_pos, n2_pos]
      for h = [0..H):
        n1_sums[b_pos, n1_pos][h] += n2_encoded[n2_idx][h]

# (2) Work on B*N rectangle.
for b_pos = [0..B):
  for n1_pos = [0..N):
    
    # Gather n1_encoded and concatenate with n1_sums.
    n1_idx = n1_sample[b_pos, n1_pos]
    for h = [0..H):
      activation[h] = n1_encoded[n1_idx][h]
      activation[h+H] = n1_sums[b_pos, n1_pos][h] / N
      
    # Apply the W_1 NN layer.
    for h = [0..H):
      for f = [0..2*H):
        temp += W_1[h][f] * activation[f];
      temp += b_1[h]
      hidden1[h] = ReLU(temp)

    # Reduce the results to src_sums.
    for h = [0..H):
      src_sums[b_pos][h] += hidden1[h]

# (3) Work on B vector.
for b_pos = [0..B):
  
  # Calculate src means.
  for h = [0..H):
    mean[h] = src_sums[b_pos][h] / N

  # Apply the W_2 layer.
  for h = [0..H):
    for f = [0..H):
      temp[h] += W_2[h][f] * mean[f];
    temp[h] += b_2[h]
    hidden0[h] = ReLU(temp[h])
          
  # Apply the W_out layer.
  for h = [0..H):
    temp += W_out[h] * hidden0[h];
  prediction[b_pos] = ReLU(temp)
