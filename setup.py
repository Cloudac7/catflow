import json

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('setup.json', 'r') as info:
    kwargs = json.load(info)

# TODO: check packages and their versions in setup.json
setup(long_description=long_description,
      long_description_content_type='text/markdown',
      include_package_data=True,
      packages=find_packages(),
      **kwargs)
