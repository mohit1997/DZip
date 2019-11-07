# DeepZip

## Description
DNA_compression using neural networks


## Requirements
0. GPU
1. python 2
2. numpy
3. sklearn
4. keras 2.2.2
5. tensorflow (gpu) 1.8

## Code
To run a compression experiment: 


## Links to the Datasets
| File | Link |
|------|------|
|webster|http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia|
|mozilla|http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia|
|h. chr20|ftp://hgdownload.cse.ucsc.edu/goldenPath/hg38/chromosomes/chr20.fa.gz|
|h. chr1|ftp://hgdownload.cse.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz|
|c.e. Gen|ftp://ftp.ensembl.org/pub/release-97/fasta/caenorhabditis_elegans/dna/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa.gz|
|ill-quality|http://bix.ucsd.edu/projects/singlecell/nbt_data.html|
|text8|http://www.mattmahoney.net/dc/textdata.html|
|enwiki9|http://www.mattmahoney.net/dc/textdata.html|
|np-bases|https://github.com/nanopore-wgs-consortium/NA12878|
|np-quality|https://github.com/nanopore-wgs-consortium/NA12878|

### Data Preparation
1. All the datasets can be downloaded using the scripts provided in the folder [Datasets](./Datasets)


Run the following command
```bash 
bash get_data.sh
```

For the PhiX virus quality scores data, a zipped file is provided directly.

### Running models
#### There are two ways of running DeepZip

##### Geting bits per symbol required (uses GPU for encoding and faster)
1. Go to [coding-gpu](./coding-gpu)
2. Place the parsed files in the directory files_to_be_compressed.
3. Run the following command

```bash 
bash get_compression_results.sh files_to_be_compressed/FILENAME
```

##### ENCODING-DECODING (uses cpu and slower)
1. Go to [encode-decode](./encode-decode)
2. Place the parsed files in the directory files_to_be_compressed.
3. Run the following command

```bash 
bash compress.sh files_to_be_compressed/FILENAME
```
