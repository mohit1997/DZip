# DZip
## improved general-purpose lossless compression based on novel neural network modeling
#### Arxiv: https://arxiv.org/abs/1911.03572
## Description
DZip is a general lossless compressor for sequential data which uses NN-based modelling combined with arithmetic coding. We refer to the NN-based model as the "combined model", as it is composed of a bootstrap model and a supporter model. The bootstrap model is trained prior to compression on the data to be compressed, and the resulting model parameters (weights) are stored as part of the compressed output (after being losslessly compressed with BSC). The combined model is adaptively trained (bootstrap model parameters are fixed) while compressing the data, and hence its parameters do not need to be stored as part of the compressed output.

## Requirements
0. GPU
1. Python3 (<= 3.6.8)
2. Numpy
3. Sklearn
4. Keras 2.2.2
5. Tensorflow (gpu) 1.14


### Download and install dependencies
Download:
```bash
git clone https://github.com/mohit1997/DZip.git
```
To set up virtual environment and dependencies (on Linux):
```bash
cd DZip
python3 -m venv tf
source tf/bin/activate
bash install.sh
```

On macOS, you need gcc compiler for running BSC which encodes the NN weights. For this, install gcc@9 using brew as follows:
```bash
brew update
brew install gcc@9
```

Then instead of `install.sh` use `install_mac.sh`
```bash
cd DZip
python3 -m venv tf
source tf/bin/activate
bash install_mac.sh
```


# USAGE
To run a compression experiment: 

### How to run DZip Compressor

User can specify to run DZip either using the combined model (default setting) or using the bootstrap model alone. Due to current limitations of the Keras platform (see "Additional Comments" below), the encoding/decoding is currently slow. Therefore, we provide a faster method to directly obtain the bits per symbol achieved by DZip, without actually compressing the file.

##### ENCODING-DECODING (uses cpu and slower)
<!-- 1. Go to [encode-decode](./encode-decode)
2. Place the parsed files in the directory files_to_be_compressed.
3. Run the following command -->

```bash 
cd encode-decode
# Compress using the combined model (default usage of DZip)
bash compress.sh FILE.txt FILE.dzip com
# Compress using only the bootstrap model
bash compress.sh FILE.txt FILE.dzip bs
# Decompress
bash decompress.sh FILE.dzip decom_FILE
# Verify successful decompression
bash compare.sh FILE.txt decom_FILE
```

##### Getting the resulting bits per symbol achieved by DZip (for both the combined model and the bootstrap only model) without compressing the file explicitly (uses GPU, faster)

```bash
cd coding-gpu
bash get_compression_results.sh files_to_be_compressed/FILE.txt
```


## Links to the Datasets
| File | Link |
|------|------|
|webster|http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia|
|mozilla|http://sun.aei.polsl.pl/~sdeor/index.php?page=silesia|
|h. chr20|ftp://hgdownload.cse.ucsc.edu/goldenPath/hg38/chromosomes/chr20.fa.gz|
|h. chr1|ftp://hgdownload.cse.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz|
|c.e. genome|ftp://ftp.ensembl.org/pub/release-97/fasta/caenorhabditis_elegans/dna/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa.gz|
|ill-quality|http://bix.ucsd.edu/projects/singlecell/nbt_data.html|
|text8|http://www.mattmahoney.net/dc/textdata.html|
|enwiki9|http://www.mattmahoney.net/dc/textdata.html|
|np-bases|https://github.com/nanopore-wgs-consortium/NA12878|
|np-quality|https://github.com/nanopore-wgs-consortium/NA12878|

##
1. Go to [Datasets](./Datasets)
2. For real datasets, run
```bash
bash get_data.sh
```
3. For synthetic datasets, run
```bash
# For generating XOR-10 dataset
python generate_data.py --data_type 0entropy --markovity 10 --file_name files_to_be_compressed/xor10.txt
# For generating HMM-10 dataset
python generate_data.py --data_type HMM --markovity 10 --file_name files_to_be_compressed/hmm10.txt
```
4. This will generate a folder named `files_to_be_compressed`. This folder contains the parsed files which can be used to recreate the results in our paper.



### Examples

To compress a synthetic sequence XOR-10. 

#### NOTE: We have already provided some sample synthetic sequences (XOR-k and HMM-k) for test runs in [coding-gpu/files_to_be_compressed](./coding-gpu/files_to_be_compressed).

#### Compress using DZip
```bash 
# Compress using Bootstrap Model
bash compress.sh files_to_be_compressed/xor10.txt xor10.dzip bs
# Compress using Combined Model
bash compress.sh files_to_be_compressed/xor10.txt xor10.dzip com
```
#### Decompress using DZip

```bash 
# Decompress
bash decompress.sh xor10.dzip decom_xor10.txt
```

#### Check if decoding is successful

```bash
bash compare.sh files_to_be_compressed/xor10.txt decom_xor10.txt
```

### Credits
The arithmetic coding is performed using the code available at [Reference-arithmetic-coding](https://github.com/nayuki/Reference-arithmetic-coding). The code is a part of Project Nayuki.

### Additional Comments

With the combined model (default setting of DZip), the compression/decompression speed is approximately 5 hours/MB due to the limitation of the [keras platform](https://keras.io/getting-started/faq/). The proposed compressor uses neural networks to model the sequence, and hence requires GPUs for training and inference. However, some of the operations are inherently non deterministic due to the underlying platform. Hence, the training and inference of the combined model is performed with CPU on a single thread, making DZip less practical for usage. In the future, we expect to bypass these limitations, and improve the compression/decompression speed significantly (10 minutes/MB).

