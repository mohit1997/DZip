wget -O nanopore_reads.fastq.gz http://s3.amazonaws.com/nanopore-human-wgs/rel3-nanopore-wgs-288418386-FAB39088.fastq.gz
gunzip nanopore_reads.fastq.gz
head -n 400000 nanopore_reads.fastq > trunc_reads.fastq
echo "Generated trunc_reads.fastq"
mkdir -p files_to_be_compressed
python parse_fastq.py
