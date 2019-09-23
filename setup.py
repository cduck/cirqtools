from setuptools import setup, find_packages
import logging
logger = logging.getLogger(__name__)

version = '0.1.0'

try:
    with open('README.md', 'r') as f:
        long_desc = f.read()
except:
    logger.warning('Could not open README.md.  long_description will be set to None.')
    long_desc = None

setup(
    name = 'cirqtools',
    packages = find_packages(),
    version = version,
    description = 'Useful classes and functions to use with Cirq circuits.',
    long_description = long_desc,
    long_description_content_type = 'text/markdown',
    author = 'Casey Duckering',
    #author_email = '',
    url = 'https://github.com/cduck/cirqtools',
    download_url = 'https://github.com/cduck/cirqtools/archive/{}.tar.gz'.format(version),
    keywords = ['cirq', 'quantum computing'],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Framework :: IPython',
        'Framework :: Jupyter',
    ],
    install_requires = [
        'cirq',
    ],
    extras_require = {
        'dev': [
            'twine',
        ]
    },
)

