#CDIUTILS: Utilities for BCDI practionners

# authors :
#              Clement Atlan, c.atlan@outlook.com
import setuptools

setuptools.setup(
      name='cdiutils',
      version='0.1.3',
      author="Clement Atlan",
      description=(
            "A python package to help Coherent Diffraction Imaging (CDI) "
            "practionionners in their analysis."
      ),
      author_email="c.atlan@outlook.com",
      scripts=[
            "scripts/bcdi_analysis.py"
      ],
      packages=setuptools.find_packages(),
      include_package_data=True
)
