import re
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("spinn/__init__.py") as fh:
    version = re.search(r"__version__ = '(.+)'", fh.read()).group(1)

setuptools.setup(
    name="spinn",
    version=version,
    author="Marek Wojciechowski",
    author_email="mwojc@p.lodz.pl",
    description="Simple Physics Informed Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrkwjc/spinn",
    packages=setuptools.find_packages(),
    keywords=['ann', 'pinn'],
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=['jax', 'jaxlib', 'numpy', 'scipy', 'networkx',
                      'optax>=0.2.3'],
)
