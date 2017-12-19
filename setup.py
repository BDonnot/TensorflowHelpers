from setuptools import setup

setup(name='TensorflowHelpers',
      version='0.1',
      description='A package to make experiments easy with tensorflow',
      url='https://github.com/BDonnot/TensorflowHelpers',
      author='Benjamin DONNOT',
      author_email='benjamin.donnot@gmail.com',
      license='GPLv3',
      packages=['TensorflowHelpers'],
      install_requires=[
          "tqdm",
          'numpy',
          'tensorflow'
      ],
      zip_safe=False)