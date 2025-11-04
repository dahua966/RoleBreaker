from setuptools import setup, find_packages

long_description = open('README.md', 'r', encoding='utf-8').read()

requirements = open('requirements.txt').read().splitlines()

setup(name='slope',
      version="0.0.1",
      install_requires=requirements,
      packages=find_packages(),
      description='Library for jailbreak test',
      python_requires='>=3.11',
      author='HuaSir',
      author_email='wzh15@nudt.edu.cn',
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent'
      ],
      license="MIT License",
      long_description=long_description,
      long_description_content_type='text/markdown')
