############### -- Union-optimized pipelined version -- ##############

value[V]

n1[V]
n1_parent[V, V]
is_connected[V, V]

n2[V]
n2_parent[V, V]

# To meet the target accuracy, we can only use the union to avoid unnecessary
# feature fetches + W_in computation. Everything else will need to allow
# redundancy.

# Prepare union'd coordinate list of sampled first-level neighbors.
for s = [0..V]:
  if in_batch[s]:
    random_sample[0..V) = []
    for r = [0..12]:
      x = rand() % |is_connected[s]|
      random_sample[x]++
    for d = [0..V):
      if is_connected[s, d]:
        for r in [0:random_sample[d]):
          n1[d] = true
          n1_parent[d, s]++
          
# Prepare union'd coordinate list of sampled second-level neighbors.
for s = [0..V):
  if n1[s]:
    for p = [0..V): # Re-sample for each parent(src) connecting to me.
      for n = [0..n1_parent[s, p]): # "
        random_sample[0..V) = []
        for r = [0..12):
          x = rand() % |is_connected[s]|
          random_sample[x]++
        for d = [0..V):
          if is_connected[s, d]:
            for r in [0:random_sample[d]):
              n2[d] = true
              n2_parent[d, s]++
    
# Digest n2 vertices.
for s = [0:V]:
  if n2[s]:
    for h = [0:H):
      for f = [0:F):
        temp[h] += W_in[h][f] * value[s][f];
      encoded[h] = ReLU(temp[h])

    for p = [0:V):
      if n2_parent[s, p] > 0: # In theory we don't need this check.
        for h = [0:H):
          n1_sums[p][h] += (encoded[h] * n2_parent[s, p]);  # Multiple potential contributions from same child.
        n1_count[p] += n2_parent[s, p];

# Digest n1 vertices, rendezvous the results with those from n2 digestion above,
# and apply the W_1 NN layer.
for s = [0:V]:
  if n1[s]:
    for h = [0:H):
      for f = [0:F):
        temp[h] += W_in[h][f] * value[s][f];
      encoded[h] = ReLU(temp[h])

    # Calculate n1 means.
    for h = [0:H):
      mean[h] = n1_sums[s][h] / n1_count[s]
      
    activation = concat(encoded, mean)
    for h = [0:H):
      for f = [0:2*H):
        temp[h] += W_1[h][f] * activation[f];
      temp[h] += b_1[h]
      hidden1[h] = ReLU(temp[h])
    
    for p = [0:V):
      if n1_parent[s, p] > 0: # In theory we don't need this check.
        for h = [0:H):
          src_sums[p][h] += (hidden1[h] * n1_parent[s, p]); # Multiple potential contributions from same child.
        src_count[p] += n1_parent[s, p];

for s = [0:V):
  if in_batch[s]:
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
      
    prediction[s] = ReLU(temp)
