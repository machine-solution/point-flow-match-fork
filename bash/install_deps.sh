if ! [[ -n "${CONDA_PREFIX}" ]]; then
    echo "You are not inside a conda environment. Please activate your environment first."
    exit 1
fi

# Install setuptools first to provide distutils (required for some packages)
pip install --upgrade setuptools wheel

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
pip install fvcore iopath
pip install --no-cache-dir pytorch3d
# Or if on cpu: 
# pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
rm -R *.egg-info

# Pypose
pip install --no-deps pypose
