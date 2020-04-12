Quickstart
==========
This guide assumes you already have Doodl installed. If you do not, head over to the :doc:`installation` section
for instructions and how to get your environment ready to go.

Basic object detection
----------------------

The most common scenario is to run the object detection process in your local development environment, 
taking advantage of the hardware that you are currently using. (For more information about offloading
the process to a backend server and what this would mean for you, check the :doc:`introduction` section.)

The following snippet of code shows how run inference on an online image:

.. code-block:: python

    from doodl import Model

    model = Model()
    detections = model.inference("http://example.com/image.jpg")

    print(detections)

The :class:`~doodl.model.Model` class is the main interface provided by Doodl. Using the 
:meth:`~doodl.model.Model.inference` function you can run object detection on the provided image and return
the list of detected objects.

By default, Doodl will use the pre-trained `faster_rcnn_inception_v2_coco <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md>`_ 
as the object detection model. The first time, the frozen inference model file will be downloaded,
and used from that point on.

Let's now see a more complex scenario where we define a different pre-trained model to run the process
on a Numpy image:

.. code-block:: python

    import numpy as np
    from PIL import Image
    from doodl import Model, Configuration

    configuration = Configuration(
        model = "ssd_mobilenet_v1_coco_2018_01_28"
    )

    image = Image.open("image.jpg")
    (width, height) = image.size
    image = np.array(image.getdata())
        .reshape((height, width, 3))
        .astype(np.uint8)

    model = Model(configuration)
    detections = model.inference(image)

    print(detections)

In this case we are loading the image using ``Pillow``, reshaping it into a Numpy array, and passing that
to the :meth:`~doodl.model.Model.inference` function to get our predictions back. We are also specifying
a different pre-trained model. You can see the full list of supported pre-trained models as part of the
:py:attr:`~doodl.configuration.Configuration.model` documentation.


Remote object detection
-----------------------

Assuming you :ref:`installed the backend server <remote-installation>`, you can run the object detection
process remotely and put that responsibility on the server. Here is a simple example, this time offloading 
the process to a backend server:

.. code-block:: python

    from doodl import Model, Configuration

    configuration = Configuration(
        endpoint = "https://example.com/doodl/inference:8080"
    )
    
    model = Model(configuration)
    detections = model.inference("http://example.com/image.jpg")

    print(detections)

This time we are using the :class:`~doodl.configuration.Configuration` class to specify the endpoint where our 
backend is listening. Assuming that the library can communicate with the endpoint, the prediction results 
would be identical as before.

Besides having the server installed, setting the :attr:`~ doodl.configuration.Configuration.endpoint` attribute is
the only change necessary to have the library work with a remote server.


Specifying a custom model
-------------------------

Beyond the supported pre-trained models, you can use any compatible custom model with Doodl. You can
simply set the :attr:`~doodl.configuration.Configuration.model` property to a URL pointing to the
frozen inference graph file in ``.tar.gz`` format: 

.. code-block:: python

    from doodl import Model, Configuration

    configuration = Configuration(
        configuration.model = "https://www.example.com/frozen_inference_graph.tar.gz"
    )
 
    model = Model(configuration)
    detections = model.inference("http://example.com/image.jpg")

    print(detections)

Assuming that the frozen inference graph can be downloaded and extracted, the library will use it
to run the object detection process.


Caching predictions
-------------------

By default, Doodl does not cache the inference results. You can change this behavior by 
using the :attr:`~doodl.configuration.Configuration.cache` property:

.. code-block:: python

    from doodl import Model, Configuration

    configuration = Configuration(cache=True)
    
    model = Model(configuration)
    detections = model.inference("http://example.com/image.jpg")

    print(detections)

Now, calling the :meth:`~doodl.model.Model.inference` function repeatedly using the same image will make use
of the cached results instead of having to run the object detection process over and over again.

