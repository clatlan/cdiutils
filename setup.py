# CDIUTILS: Utilities for BCDI practionners

# authors :
#              Clement Atlan, c.atlan@outlook.com
import setuptools

setuptools.setup(
      name='cdiutils',
      version='0.1.3',
      author="Clement Atlan",
      description=(
            "A python package to help Coherent Diffraction Imaging (CDI) "
            "practitioners in their analysis"
      ),
      author_email="c.atlan@outlook.com",
      scripts=[
            "scripts/analyze_bcdi_data.py",
            "scripts/prepare_bcdi_notebook.py",
            "scripts/prepare_parameter_files.py"
      ],
      packages=setuptools.find_packages(),
      data_files=[
            (
                  '',
                  [
                        "cdiutils/processing/pynx-id01cdi_template.slurm",
                        "cdiutils/examples/analyze_bcdi_data.ipynb",
                  ]
            )
      ],
      include_package_data=True,
      # package_data={
      #       "cdiutils": [
      #             "cdiutils/processing/pynx-id01cdi_template.slurm",
      #             "scripts/analyze_bcdi_data.py",
      #             "scripts/analyze_bcdi_data.ipynb",
      #             "scripts/prepare_bcdi_notebook.py"
      #       ]
      # },
      url="https://github.com/clatlan/cdiutils",
      python_requires=">=3.8, <3.10",
      install_requires=[
            "bcdi @ git+https://github.com/carnisj/bcdi.git@6a2d2515d9b7e3a5fc7b6e7ecc13da712e071486",
            "colorcet>=3.0.0",
            "h5py>=3.6.0",
            "hdf5plugin>=3.2.0",
            "matplotlib>=3.5.2",
            "numpy>=1.23.5",
            "pandas>=1.4.2",
            "paramiko>=2.12.0",
            "PyYAML>=6.0",
            "ruamel.yaml>=0.17.21",
            "ruamel.yaml.clib>=0.2.7",
            "scikit-image>=0.19.2",
            "scikit-learn>=1.1.3",
            "scipy>=1.8.0",
            "seaborn>=0.12.1",
            "silx>=1.0.0",
            # "sklearn>=0.0.post1",
            "vtk>=9.1.0",
            "xrayutilities>=1.7.3"
      ]
)
