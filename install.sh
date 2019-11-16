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
      setuptools==41.0.0 \
      scipy \
      scikit-learn

cd libbsc && make
cp bsc ../encode-decode/
cp bsc ../coding-gpu/
