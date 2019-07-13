mkdir data
wget 'ftp://ftp.ensembl.org/pub/release-97/fasta/caenorhabditis_elegans/dna/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa.gz' -O ./data/celegchr.fa.gz
gunzip ./data/celegchr.fa.gz