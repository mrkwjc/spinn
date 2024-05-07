import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spinn",
    version="0.1.0",
    author="Marek Wojciechowski",
    author_email="mwojc@p.lodz.pl",
    description="Simple Physics Informed Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrkwjc/spinn",
    packages=setuptools.find_packages(),
    keywords=['ann', 'pinn'],
    license='LGPL-2.1',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['jax', 'numpy', 'scipy', 'networkx'],
)
