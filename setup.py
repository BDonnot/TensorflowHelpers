from setuptools import setup

setup(name='TensorflowHelpers',
      version='0.1',
      description='The funniest joke in the world',
      url='https://github.com/BDonnot/TensorflowHelpers',
      author='Benjamin DONNOT',
      author_email='benjamin.donnot@gmail.com',
      license='GPLv3',
      packages=['TensorflowHelpers'],
      install_requires=[
          'numpy',
          'tensorflow'
      ],
      zip_safe=False)