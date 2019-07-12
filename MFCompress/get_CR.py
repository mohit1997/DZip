import os
import argparse
import sys
import csv

def getSize(filename):
    st = os.stat(filename)
    return st.st_size

parser = argparse.ArgumentParser()
parser.add_argument('-i', action='store', default=None,
                    dest='data',
                    help='choose sequence file')

parser.add_argument('-csv', action='store', default="results.csv",
                    dest='results',
                    help='choose csv file')

parser.add_argument('-mode', action='store', default=3,
                    dest='mode',
                    help='choose Compression Mode 1/2/3')

arguments = parser.parse_args()

if not os.path.isfile(arguments.results):
	with open(arguments.results, 'a') as myFile:
		writer = csv.writer(myFile)
		writer.writerow(["File Name", "Original Size", "Compressed Size", "bpb"])

if arguments.data[-4:] == ".txt":
	print("Adding header")
	filename = os.path.basename(arguments.data)[:-4] + ".fa"
	os.system('echo ">" > header.fa')
	os.system('cat {} {} > {}'.format('header.fa', arguments.data, filename))
elif arguments.data[-3:] == ".fa":
	pass

with open(arguments.data) as fp:
    data = fp.read()

totallength = len(data)

os.system('./MFCompressC -{} {}'.format(arguments.mode, filename))
totalbytes = getSize(filename + ".mfc")
bpb = totalbytes*8.0/totallength
print(bpb)

with open(arguments.results, 'a') as myFile:
	writer = csv.writer(myFile)
	writer.writerow([os.path.splitext(filename)[0], getSize(arguments.data), totalbytes, bpb])


os.system("rm {}".format(filename))
os.system("rm {}".format(filename+".mfc"))
os.system("rm {}".format("header.fa"))


