if ! [[ -n "${CONDA_PREFIX}" ]]; then
    echo "You are not inside a conda environment. Please activate your environment first."
    exit 1
fi

# Set default COPPELIASIM_ROOT if not defined
if ! [[ -n "${COPPELIASIM_ROOT}" ]]; then
    
    echo "Setting environment variables in conda environment..."
    conda env config vars set COPPELIASIM_ROOT=${COPPELIASIM_ROOT}
    conda env config vars set LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}
    conda env config vars set QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}
    echo "Environment variables set. Please run 'conda deactivate' and 'conda activate pfp_env' to apply changes."
    echo "Or export them manually for this session:"
    echo "  export COPPELIASIM_ROOT=${COPPELIASIM_ROOT}"
    echo "  export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:\${COPPELIASIM_ROOT}"
    echo "  export QT_QPA_PLATFORM_PLUGIN_PATH=\${COPPELIASIM_ROOT}"
    export COPPELIASIM_ROOT=${COPPELIASIM_ROOT}
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}
    export QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}
fi

# Download Coppelia sim if not present
if ! [[ -e $COPPELIASIM_ROOT ]]; then
    wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
    mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
    rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
fi

# Install PyRep and RLBench
pip install -r https://raw.githubusercontent.com/stepjam/PyRep/master/requirements.txt
pip install git+https://github.com/stepjam/PyRep.git
pip install git+https://github.com/stepjam/RLBench.git

# Install gymnasium (required by RLBench but may not be installed automatically)
pip install gymnasium
