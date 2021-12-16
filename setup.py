from setuptools import setup

setup(
    name='miko_tasker',
    version='0.0.1',
    packages=['miko_tasker'],
    install_requires=[
        'importlib; python_version >= "3.6"',
        'ase',
        'dpdata',
        'pymatgen',
        'numpy'
    ],
)
