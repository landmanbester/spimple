from setuptools import setup, find_packages
import spimple

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
                'omegaconf',
                'katbeam',
                'pytest >= 6.2.2',
                'pyscilog >= 0.1.2',
                'codex-africanus[complete]',
                'dask-ms[xarray]',
            ]


setup(
     name='spimple',
     version=spimple.__version__,
     author="Landman Bester",
     author_email="lbester@sarao.ac.za",
     description="Radio astronomy image post-processing tools",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/landmanbester/spimple",
     packages=find_packages(),
     python_requires='>=3.7',
     install_requires=requirements,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     entry_points={
                    'console_scripts': [
                    'spimple-imconv=spimple.apps.image_convolver:image_convolver',
                    'spimple-spifit=spimple.apps.spi_fitter:spi_fitter',
                    'spimple-binterp=spimple.apps.power_beam_maker:power_beam_maker'
                    ]
     }
     ,
 )
