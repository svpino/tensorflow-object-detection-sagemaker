ARG ARCHITECTURE=1.15.0-gpu
FROM tensorflow/tensorflow:${ARCHITECTURE}-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget zip unzip git ca-certificates curl nginx

# We need to install Protocol Buffers (Protobuf). Protobuf is Google's language and 
# platform-neutral, extensible mechanism for serializing structured data. To make sure you
# are using the most updated code, replace the linked release below with the latest version
# available on the Git repository.
#RUN curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protoc-3.11.4-linux-x86_64.zip
#RUN unzip protoc-3.11.4-linux-x86_64.zip -d protoc3
#RUN mv protoc3/bin/* /usr/local/bin/
#RUN mv protoc3/include/* /usr/local/include/

# Let's add the folder that we are going to be using to install all of our machine 
# learning-related code to the PATH. This is the folder used by SageMaker to find and run
# our code.
ENV PATH="/opt/ml/code:${PATH}"
RUN mkdir -p /opt/ml/code
WORKDIR /opt/ml/code

# Let's copy all of the pre-trained models that should be pre-installed with the
# container.
RUN mkdir -p /opt/ml/pretrained
COPY /pretrained /opt/ml/pretrained

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install cython
RUN pip install -r requirements.txt

# Let's now download Tensorflow from the official Git repository and install Tensorflow Slim.
#RUN git clone https://github.com/tensorflow/models/ tensorflow-models
#RUN pip install -e tensorflow-models/research/slim

# We can now install the Object Detection API, also part of the Tensorflow repository. We are
# going to change the working directory so we can do this easily.
#WORKDIR /opt/ml/code/tensorflow-models/research
#RUN protoc object_detection/protos/*.proto --python_out=.
#RUN python setup.py build
#RUN python setup.py install

# If you are interested in using COCO evaluation metrics, you can tun the following commands
# to add the necessary resources to your Tensorflow installation.
#RUN git clone https://github.com/cocodataset/cocoapi.git
#WORKDIR /opt/ml/code/tensorflow-models/research/cocoapi/PythonAPI
#RUN make 
#RUN cp -r pycocotools /opt/ml/code/tensorflow-models/research/

# Let's put the working directory back to where it needs to be, copy all of our code, and
# update the PYTHONPATH to include the newly installed Tensorflow libraries.
WORKDIR /opt/ml/code
COPY /code /opt/ml/code

#RUN pip install doodl[tensorflow]
RUN pip install doodl_tensorflow-0.1.0.0-py3-none-any.whl
RUN pip install doodl-0.1.0.0-py3-none-any.whl[tensorflow]

#ENV PYTHONPATH=${PYTHONPATH}:tensorflow-models/research:tensorflow-models/research/slim:tensorflow-models/research/object_detection
#ENV PYTHONPATH=${PYTHONPATH}:tensorflow-models/research:tensorflow-models/research/slim:tensorflow-models/research/object_detection

RUN chmod +x train
RUN chmod +x serve