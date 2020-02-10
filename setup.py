from setuptools import setup, find_packages
packages = find_packages()

setup(name='drlnd',
      version='0.0.1',
      description='Udacity DRLND Main package',
      url='http://github.com/yycho0108/drlnd',
      author='Jamie Cho',
      author_email='jchocholate@gmail.com',
      license='MIT',
      packages=packages,
      zip_safe=False,
      scripts=[],
      install_requires=[
          'cho-util',
          'numpy',
          'torch',
          'tqdm',
          'gym',
          'hydra-core',
      ],
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],
      )
