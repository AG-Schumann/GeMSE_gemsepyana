from setuptools import setup

def readme():
  with open('README.rst') as f:
    return f.read()

setup(name='gemsepyana',
  version='0.1',
  description='Uproot based code to handle CAEN-type data for GeMSE analysis.',
  long_description=readme(),
  url='http://github.com/AG-Schumann/gemsedata',
  author='Sebastian Lindemann',
  author_email='sebastian.lindemann@physik.uni-freiburg.de',
  license='MIT',
  packages=['gemsepyana'],
  install_requires=[
    'numpy', 
    'uproot',
    'matplotlib',
    ],
  include_package_data=True,
  entry_points={
    'console_scripts': [
      'annotate_histogram=gemsepyana.annotate_calibrated_histogram_uproot:main',
      'annotate_using_bat_inputs=gemsepyana.annotate_using_bat_inputs:main',
      'plot_spectrum_paper_style=gemsepyana.plot_spectrum_paper_style_script:main',
      ],
    },
  zip_safe=False)
