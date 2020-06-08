from setuptools import setup, find_packages

def read_readme():
    with open("README.md") as fp:
        long_desc = fp.read()
    return long_desc

setup(
    name = "tinygrad",
    version = "1.0.0",
    author = "Pranjal Datta",
    description = ("A simple, basic autodiff engine written to learn the inner workings of autograd!"),
    license = "MIT",
    long_description = read_readme(),
    url = "https://github.com/pranjaldatta/tinygrad",

    packages = find_packages(),
   
)