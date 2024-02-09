import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name="tgnnu",
    version="3.0.5",

    author='Soroush H. Zargarbashi',
    author_email='soroushzargar@gmail.com',
    url='https://github.com/soroushzargar/TorchGNN-Utils.git',

    description="Toolkits for easier work with PyTorch Geometric",

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=setuptools.find_packages(),

    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    license='License :: OSI Approved :: MIT License',
)
