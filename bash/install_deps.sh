if ! [[ -n "${CONDA_PREFIX}" ]]; then
    echo "You are not inside a conda environment. Please activate your environment first."
    exit 1
fi

# setuptools<74 keeps distutils.msvccompiler so packages like av build correctly
pip install 'setuptools<74' wheel

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
pip install fvcore iopath

# PyTorch3D: not on PyPI; use conda (recommended) or build from source
if conda install -y -c fvcore -c conda-forge pytorch3d 2>/dev/null; then
    echo "pytorch3d installed via conda"
else
    echo "Trying pytorch3d from pytorch3d channel..."
    conda install -y -c pytorch3d pytorch3d 2>/dev/null || echo "WARNING: pytorch3d not installed. Install manually if needed."
fi

# Optional: av for video (needs system: pkg-config libavformat-dev libavcodec-dev libavutil-dev libswscale-dev)
# pip install av==8.1.0 --no-build-isolation  # or: pip install -e ".[video]" after installing system libs

pip install -e .
rm -Rf *.egg-info 2>/dev/null || true

# Pypose
pip install --no-deps pypose
