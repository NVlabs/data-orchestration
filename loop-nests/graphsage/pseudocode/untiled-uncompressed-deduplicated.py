############### -- Union-optimized pipelined version -- ##############

# Parameters:
V = 1M  # number of vertices
F = 32  # feature size per vertex
H = 64  # hidden layer size
B = 256 # batch size
N = 12  # neighbor samples per vertex

# Inputs:
float features[V][F]
bool  is_connected[V, V]
bool  n1_sample[V][B][V][N]
bool  n2_sample[V][B][V][N][V][N]

# Intermediate tensors:
bool  n1_dedup[V]
bool  n2_dedup[V]
float n1_sums[V][B][V][N][H]
int   n1_count[V][B][V][N]
float src_sums[V][B][H]
int   src_count[V][B]

# Output:
float prediction[V][B]

# A coordinate-tensor of size [X] in the compressed space is represented as a boolean
# tensor of size [V][X] in the uncompressed space. The boolean (v, x) tells us whether
# v was present in the x-th position in the vector [X]. This representation allows for
# duplicate entries in the [X] vector.

# Similarly. a coordinate-tensor of size [X][Y] in the compressed space is represented
# as a boolean tensor of size [V][X][V][Y] in the uncompressed space.

batch: bool [1M][256]
n1_sample: bool [1M][256][1M][12]
n2_sample: bool [1M][256][1M][12][1M][12]

# Previously, we were using a set of "parent" tensors to locate parent notes to propagate values to.
# However, in this dense representation, we can simply reuse the n1_sample and n2_sample tensors.

# for s = [0..V]:
#   if in_batch[s]:
#     random_sample[0..V) = []
#     for r = [0..12]:
#       x = rand() % |is_connected[s]|
#       random_sample[x]++

# for s = [0..V):
#   if n1_dedup[s]:
#     for p = [0..V): # Re-sample for each parent(src) connecting to me.
#       for n = [0..n1_parent[s, p]): # "
#         random_sample[0..V) = []
#         for r = [0..12):
#           x = rand() % |is_connected[s]|
#           random_sample[x]++

# To meet the target accuracy, we can only use deduplication to avoid redundant
# feature fetches + W_in computation. Everything else will need to allow
# redundancy.

# Dedup n1 neighbors and prepare parent links.
for s = [0..V]:
  for b = [0..B): # B = 256. b identifies which of the 256 positions in the batch 's' belongs to.
    if batch[s, b]:
      for n1 = [0..V):        
        for n1_pos = [0..N): # N = 12. n1_pos identifies which of the 12 positions in s's neighbor-sample vector n1 belongs to.      
          if n1_sample[s, b, n1, n1_pos]:
            n1_dedup[n1] = true
            n1_parent[n1, b, s]++

# Dedup n2 neighbors and prepare parent links.
for s = [0..V):
  for b = [0..B): # B = 256. b identifies which of the 256 positions in the batch 's' belongs to.
    if batch[s, b]:
      for n1 = [0..V):
        for n1_pos = [0..N): # N = 12. n1_pos identifies which of the 12 positions in s's neighbor-sample vector n1 belongs to.      
          if n1_sample[s, b, n1, n1_pos]:
            for n2 = [0..N):
              for n2_pos = [0..N): # N = 12. n2_pos identifies which of the 12 positions in n1's neighbor-sample vector n2 belongs to.      
                if n2_sample[s, b, n1, n1_pos, n2, n2_pos]:
                  n2_dedup[n2] = true
                  n2_parent[n2, n1_pos, n1]++

# Digest n2 vertices.
for n2 = [0:V]:
  if n2_dedup[n2]:
    for h = [0:H):
      for f = [0:F):
        temp[h] += W_in[h][f] * features[n2][f];
      encoded[h] = ReLU(temp[h])

    # I have an encoded value for each distinct n2. Now
    # propagate this value to whoever needed it.
    for s = [0..V]:
      for b = [0..B): # B = 256. b identifies which of the 256 positions in the batch 's' belongs to.
        if batch[s, b]:
          for n1 = [0..V):        
            for n1_pos = [0..N): # N = 12. n1_pos identifies which of the 12 positions in s's neighbor-sample vector n1 belongs to.      
              if n1_sample[s, b, n1, n1_pos]:
                for h = [0:H):
                  n1_sums[s, b, n1, n1_pos][h] += encoded[h];  # Multiple potential contributions from same child.
                n1_count[s, b, n1, n1_pos]++;

# Digest n1 vertices, rendezvous the results with those from n2 digestion above,
# and apply the W_1 NN layer.
for n1 = [0:V]:
  if n1_dedup[n1]:
    for h = [0:H):
      for f = [0:F):
        temp[h] += W_in[h][f] * features[n1][f];
      encoded[h] = ReLU(temp[h])

    # I have an encoded value for each distinct n1. Now
    # propagate this value to whoever needed it.
    for s = [0..V]:
      for b = [0..B): # B = 256. b identifies which of the 256 positions in the batch 's' belongs to.
        if batch[s, b]:
          for n1_pos = [0..N): # N = 12. n1_pos identifies which of the 12 positions in s's neighbor-sample vector n1 belongs to.      
            # I'm now working with a unique (s,n1,n1_pos) tuple.

            # Calculate n1 means.
            for h = [0:H):
              mean[h] = n1_sums[s, b, n1, n1_pos][h] / n1_count[s, b, n1, n1_pos]

            # Concatenate and apply the next NN layer.
            activation = concat(encoded, mean)
            for h = [0:H):
              for f = [0:2*H):
                temp[h] += W_1[h][f] * activation[f];
              temp[h] += b_1[h]
              hidden1[h] = ReLU(temp[h])

            # Propagate the results to src_sums.
            for h = [0:H):
              src_sums[s, b][h] += hidden1[h]
            src_count[s, b]++
            
# Final steps.
for s = [0..V):
  for b = [0..B): # B = 256. b identifies which of the 256 positions in the batch 's' belongs to.
    if batch[s, b]:
      
      # Calculate src means.
      for h = [0:H):
        mean[h] = src_sums[s, b][h] / src_count[s, b]

      for h = [0:H):
        for f = [0:H):
          temp[h] += W_2[h][f] * mean[f];
        temp[h] += b_2[h]
        hidden0[h] = ReLU(temp[h])

      for h = [0:H):
        temp += W_out[h] * hidden0[h];
      
      prediction[s, b] = ReLU(temp)
