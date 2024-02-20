# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2020, Multi-Agent Particle Environment'
__credits__ = ['Rolando Fernandez', 'OpenAI']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rfernandez@utexas.edu'
__status__ = 'Dev'

setup(name='multiagent_particle_env',
      version='0.0.1',
      description='Multi-Agent Goal-Driven Particle Environment',
      url='',
      author='Rolando Fernandez',
      author_email='rfernandez@utexas.edu',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['colorama', 'gym', 'numpy', 'numpy-stl', 'pyaml', 'pyglet==1.5.27', 'six'])
