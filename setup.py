from setuptools import find_packages
import setuptools


setuptools.setup(
    name="encoder-image-torch",
    version="0.0.1",
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description="Executor encodes images into d-dimensional vector space using neural network.",
    url="https://github.com/jina-ai/encoder-image-torch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where='.', include=['jinahub.*']),
    install_requires=open("requirements.txt").readlines(),
    python_requires=">=3.7"
)
