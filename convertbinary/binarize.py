def strToBinary(s, dic): 
	bin_conv = [] 

	for c in s: 
		bin_conv.append(dic[c]) 
		
	return (''.join(bin_conv)) 


import sys
import numpy as np
import json
import string
import itertools

input_file = "text8"
output_file = input_file + "_bin"
param_file = "map_dic"

with open(input_file) as fp:
    data = fp.read()

vals = list(set(data))
vals.sort()
print(vals, len(vals))

alphabetsize = 4
codesize = int(np.ceil(np.log(len(vals))/np.log(alphabetsize)))

letters = string.ascii_lowercase[:alphabetsize]
keywords = [''.join(i) for i in itertools.product(letters, repeat=codesize)][:len(vals)]
print(keywords, len(keywords))

assert len(keywords) == len(vals)

map_dic = {}
for i, l in enumerate(vals):
	map_dic[l] = keywords[i]

with open(param_file, 'w') as f:
    json.dump(map_dic, f, indent=4)

f = open(output_file,"w")

data = strToBinary(data, map_dic)

f.write(data)
f.close()

