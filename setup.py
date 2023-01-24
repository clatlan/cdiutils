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
            "scripts/bcdi_analysis.py"
      ],
      packages=setuptools.find_packages(),
      include_package_data=True,
      url="https://github.com/clatlan/cdiutils",
      install_requires=[
            "bcdi @ git+https://github.com/carnisj/bcdi.git@6a2d2515d9b7e3a5fc7b6e7ecc13da712e071486",
            "colorcet>=3.0.0",
            "h5py>=3.6.0",
            "hdf5plugin>=3.2.0",
            "matplotlib>=3.5.2",
            "numpy>=1.22.3",
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
            "sklearn>=0.0.post1",
            "vtk==9.1.0",
            "xrayutilities>=1.7.3"
      ]
)
