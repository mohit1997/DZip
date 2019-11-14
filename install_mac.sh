pip install --upgrade pip
pip install \
    tensorflow-gpu==1.14
pip install tqdm
pip install \
      keras \
      argparse \
      pandas \
      h5py \
      "numpy<1.17" \
      scipy \
      scikit-learn

cd libbsc && make CC=g++-9
cp bsc ../encode-decode/
cp bsc ../coding-gpu/
