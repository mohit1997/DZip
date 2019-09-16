from itertools import ifilter, islice

def readFastq(fastqfile):
    "parse a fastq-formatted file, yielding a (header, sequence, quality) tuple"
    fastqiter = (l.strip('\n') for l in fastqfile)  # strip trailing newlines 
    fastqiter = ifilter(lambda l: l, fastqiter)  # skip blank lines
    while True:
        fqlines = list(islice(fastqiter, 4))
        if len(fqlines) == 4:
            header1,seq,header2,qual = fqlines
	    print(header1, seq, header2, qual)
        elif len(fqlines) == 0:
            raise StopIteration
        else:
            raise EOFError("Failed to parse four lines from fastq file!")

        if header1.startswith('@') and header2.startswith('+'):
            yield header1[1:], seq, qual
	    break
        else:
            raise ValueError("Invalid header lines: %s and %s" % (header1, header2))


fname = 'first_400000.fastq'
with open(fname, 'r') as outfile:
	out = outfile.read()

# output = readFastq(out)
# lis = []
# for v1, v2, v3 in output:
# 	lis.append(v2)

# print(len(lis))

class ParseFastQ(object):
    """Returns a read-by-read fastQ parser analogous to file.readline()"""
    def __init__(self,filePath,headerSymbols=['@','+']):
        """Returns a read-by-read fastQ parser analogous to file.readline().
        Exmpl: parser.next()
        -OR-
        Its an iterator so you can do:
        for rec in parser:
            ... do something with rec ...
        rec is tuple: (seqHeader,seqStr,qualHeader,qualStr)
        """
        if filePath.endswith('.gz'):
            self._file = gzip.open(filePath)
        else:
            self._file = open(filePath, 'rU')
        self._currentLineNumber = 0
        self._hdSyms = headerSymbols
        
    def __iter__(self):
        return self
    
    def next(self):
        """Reads in next element, parses, and does minimal verification.
        Returns: tuple: (seqHeader,seqStr,qualHeader,qualStr)"""
        # ++++ Get Next Four Lines ++++
        elemList = []
        for i in range(4):
            line = self._file.readline()
            self._currentLineNumber += 1 ## increment file position
            if line:
                elemList.append(line.strip('\n'))
            else: 
                elemList.append(None)
        
        # ++++ Check Lines For Expected Form ++++
        trues = [bool(x) for x in elemList].count(True)
        nones = elemList.count(None)
        # -- Check for acceptable end of file --
        if nones == 4:
            raise StopIteration
        # -- Make sure we got 4 full lines of data --
        assert trues == 4,\
               "** ERROR: It looks like I encountered a premature EOF or empty line.\n\
               Please check FastQ file near line number %s (plus or minus ~4 lines) and try again**" % (self._currentLineNumber)
        # -- Make sure we are in the correct "register" --
        assert elemList[0].startswith(self._hdSyms[0]),\
               "** ERROR: The 1st line in fastq element does not start with '%s'.\n\
               Please check FastQ file near line number %s (plus or minus ~4 lines) and try again**" % (self._hdSyms[0],self._currentLineNumber) 
        assert elemList[2].startswith(self._hdSyms[1]),\
               "** ERROR: The 3rd line in fastq element does not start with '%s'.\n\
               Please check FastQ file near line number %s (plus or minus ~4 lines) and try again**" % (self._hdSyms[1],self._currentLineNumber) 
        # -- Make sure the seq line and qual line have equal lengths --
        assert len(elemList[1]) == len(elemList[3]), "** ERROR: The length of Sequence data and Quality data of the last record aren't equal.\n\
               Please check FastQ file near line number %s (plus or minus ~4 lines) and try again**" % (self._currentLineNumber) 
        
        # ++++ Return fatsQ data as tuple ++++
        return tuple(elemList)

import sys
import numpy as np
import json
import argparse
import os

input_file = fname
base_name = os.path.basename(input_file)
param_file = "params_" + os.path.splitext(base_name)[0]
output_file = os.path.splitext(base_name)[0]

parser = ParseFastQ(fname)
data = ''
for tup in parser:
	data += tup[1] + 'E'

print("Seq Length {}".format(len(data)))
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
print(list(integer_encoded[:10, 0]))
print(data[:10])

np.save(output_file, integer_encoded)
	
