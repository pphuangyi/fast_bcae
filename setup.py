from setuptools import setup

setup(
    name = "RTID",
    version = "0.0.1.dev",
    author = "TBD",
    author_email = "TBD",
    description = ("New ideas for TPC data compression and track reconstruction"),
    license = "MIT",
    keywords = "BSD 3-Clause 'New' or 'Revised' License",
    # url = "https://github.com/BNL-DAQ-LDRD/NeuralCompression",
    packages=['rtid',],
    long_description="",
    install_requires=[
        "torch",
        "numpy",
        "pandas"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: BSD 3-Clause 'New' or 'Revised' License",
    ],
)
