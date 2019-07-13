#/bin/bash
fasta_dir="data"
data_dir="files_to_be_compressed"
mkdir -p $data_dir;

for f in $fasta_dir/*.fa
do
    echo "filename: "$f
    s=${f##*/}
    basename=${s%.*}
    echo $basename
    
    output_file="$data_dir/$basename.txt"
    sed '/>/d' $f | tr -d '\n' | tr '[:lower:]' '[:upper:]' > $output_file
    echo "- - - - - "
done
