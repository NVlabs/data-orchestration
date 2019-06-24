############### -- Un-optimized, imperative version -- ##############

# Parameters:
V = 1M  # number of vertices
F = 32  # feature size per vertex
H = 64  # hidden layer size
B = 256 # batch size
N = 12  # neighbor samples per vertex

# A coordinate-tensor of size [X] in the compressed space is represented as a boolean
# tensor of size [V][X] in the uncompressed space. The boolean (v, x) tells us whether
# v was present in the x-th position in the vector [X]. This representation allows for
# duplicate entries in the [X] vector.

# Similarly. a coordinate-tensor of size [X][Y] in the compressed space is represented
# as a boolean tensor of size [V][X][V][Y] in the uncompressed space.

# Inputs:
float features[V][F]
bool  is_connected[V, V]

# Intermediate tensors:
bool  batch[V][B]
bool  n1_sample[V][B][V][N]
bool  n2_sample[V][B][V][N][V][N]

bool  n1_dedup[V]
bool  n2_dedup[V]
float n1_sums[V][B][V][N][H]
int   n1_count[V][B][V][N]
float src_sums[V][B][H]
int   src_count[V][B]

# Output:
float prediction[V][B]

# Pre-generate (in un-simulated code) batch, n1 and n2 samples.
for b = [0..B):
  s = rand() % V
  batch[s, b] = true
  
  for n1_pos = [0..N):
    n1_idx = rand() % num_neibs[s]
    n1 = neib_lookup[n1_idx]
    n1_sample[s, b, n1, n1_pos] = true
    
    for n2_pos = [0..N):
      n2_idx = rand() % num_neibs[n1]
      n2 = neib_lookup[n2_idx]
      n2_sample[s, b, n1, n1_pos, n2, n2_pos] = true

# Go!
for s = [0..V):
  for b = [0..B): # b identifies which of the 256 positions in the batch 's' belongs to.
    if batch[s, b]:
      
      # Look for n1 vertices.
      for n1 = [0..V):
        for n1_pos = [0..N): # n1_pos identifies which of the 12 positions in s's neighbor-sample vector n1 belongs to.      
          if n1_sample[s, b, n1, n1_pos]:
            
            # "Recurse" into n2 vertices.
            for n2 = [0..N):
              for n2_pos = [0..N): # n2_pos identifies which of the 12 positions in n1's neighbor-sample vector n2 belongs to.      
                if n2_sample[s, b, n1, n1_pos, n2, n2_pos]:
                  for h = [0:H):
                    for f = [0:F):
                      temp[h] += W_in[h][f] * value[n2][f]
                    encoded[h] = ReLU(temp[h])
                    n1_sums[n1][h] += encoded[h]
                  n1_count[n1]++

            # Digest n1 vertices, rendezvous the results with those from n2 digestion above,
            # and apply the W_1 NN layer.
            for h = [0:H):
              for f = [0:F):
                temp[h] += W_in[h][f] * value[n1][f]
              encoded[h] = ReLU(temp[h])

            # Calculate n1 means.
            for h = [0:H):
              mean[h] = n1_sums[n1][h] / n1_count[n1]

            activation = concat(encoded, mean)
            for h = [0:H):
              for f = [0:2*H):
                temp[h] += W_1[h][f] * activation[f];
              temp[h] += b_1[h]
              hidden1[h] = ReLU(temp[h])
              src_sums[s][h] += hidden1[h];
            src_count[s]++;
  
      # Calculate src means.
      for h = [0:H):
        mean[h] = src_sums[s][h] / src_count[s]

      for h = [0:H):
        for f = [0:H):
          temp[h] += W_2[h][f] * mean[f];
        temp[h] += b_2[h]
        hidden0[h] = ReLU(temp[h])

      for h = [0:H):
        temp += W_out[h] * hidden0[h];
      prediction[s, b] = ReLU(temp)
      
