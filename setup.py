import os
import pkg_resources
from setuptools import setup, find_packages

setup(
    name = 'perceiver-pytorch',
    packages = find_packages(exclude=["somepkg_names"]),
    version = '0.0.1',         # major, minor, patch version
    license='MIT',
    description = 'Boostraping multi-modality in self-supervised learning framework',
    author = 'HHRI SSL-group',
    author_email = 'tvnhiemhmus@g.ncu.edu.tw',
    url = 'https://github.com/TranNhiem/Boostraping_Multi_modalities_SSL',
    keywords = [
        'multi-modality',
        'deep learning',
        'transformer',
        'self-supervised learning'
    ],
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    classifiers=[
        'Development Status :: 1 - alpha',
        'Intended Audience :: Researcher',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8'
    ],
    extras_require={}
)