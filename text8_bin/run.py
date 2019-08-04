import sys
import numpy as np
import json

input_file = "text8_bin"
param_file = "params"
output_file = "output"

with open(input_file) as fp:
    data = fp.read()

print(len(data))
vals = list(set(data))
vals.sort()
print(vals)

char2id_dict = {c: i for (i,c) in enumerate(vals)}
id2char_dict = {i: c for (i,c) in enumerate(vals)}

params = {'char2id_dict':char2id_dict, 'id2char_dict':id2char_dict}
with open(param_file, 'w') as f:
    json.dump(params, f, indent=4)

print(char2id_dict)
print(id2char_dict)

out = [char2id_dict[c] for c in data]
integer_encoded = np.array(out)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
print(integer_encoded[:10])
print(data[:10])

np.save(output_file, integer_encoded)
