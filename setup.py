from setuptools import setup
from setuptools import find_packages

required_packages = []


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='mcq-checker',
      version='0.1',
      description='MCQ Checker',
      url='https://github.com/n1amr/mcq-checker',
      author='Amr Alaa',
      author_email='n1amr1@gmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires=required_packages,
      entry_points={
          'console_scripts': [
              'mcq-checker = mcq-checker.__main__:main']},
      zip_safe=False, )
