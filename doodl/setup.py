from setuptools import setup, find_packages


setup(
    name="doodl",
    version="0.1.0.0",
    description="Object Detection Library",
    license="Apache License 2.0",
    url="https://github.com/svpino/doodl",
    author="Santiago L. Valdarrama",
    author_email="svpino@gmail.com",
    python_requires=">=3.6, <4",
    packages=find_packages(exclude=["tests"]),
    extras_require={
        "tensorflow":  ["doodl_tensorflow"],
    },
    install_requires=[
        "setuptools>=41.0.0",
        "numpy",
        "Pillow",
        "boto3",
        "requests",
        "six"
    ],
    zip_safe=False,
)
