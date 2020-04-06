import glob
import os
import shutil

from setuptools import setup, find_packages


CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))


def _install_protobuf_files():
    """
    Copy pre-compiled protobuf files into tensorflow distribution.
    """
    path = os.path.join(
        CURRENT_DIRECTORY,
        "tensorflow-object-detection/research/object_detection/protos/",
    )
    for filename in glob.glob(os.path.join(CURRENT_DIRECTORY, "pb2", "*.*")):
        shutil.copy(filename, path)


_install_protobuf_files()

setup(
    name="todl",
    version="0.1.1",
    description="Tensorflow Object Detection Library",
    license="Apache License 2.0",
    url="https://github.com/svpino/todl",
    author="Santiago L. Valdarrama",
    author_email="svpino@gmail.com",
    python_requires=">=3.6, <4",
    packages=find_packages(
        where="tensorflow-object-detection/research",
        include=["object_detection", "object_detection.*"],
    )
    + find_packages(
        where="tensorflow-object-detection/research/slim", include=["datasets.*"]
    )
    + find_packages(exclude=["tests"]),
    package_dir={
        "object_detection": "tensorflow-object-detection/research/object_detection",
    },
    install_requires=[
        "setuptools>=41.0.0",
        "tensorflow==1.15",
        "cython",
        "numpy",
        "Pillow",
        "contextlib2",
        "matplotlib",
        "boto3",
        "requests"
    ],
    zip_safe=False,
)
