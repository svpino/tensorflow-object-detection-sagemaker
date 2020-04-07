Quickstart
==========
This page assumes you already have Doodl installed. If you do not, head over to the :doc:`installation` section.


Object detection in your local environment
------------------------------------------

The most common scenario is to run the object detection process in your local development environment, 
taking advantage of the hardware that you are currently using.

Running the process directly in your development environment means that you eliminate the latency introduced 
by calling a remote endpoint. When processing a lot of content, the communication time adds up quickly, so 
this approach undoubtedly results in increased performance.

The downside is that you need to install all of the plumbing necessary to run the object detection process. 
Although Doodl makes the installation process very simple, it forces the installation of Tensorflow 1.15 
in your environment. This may be a problem if you want to use Tensorflow 2.x in different components of your 
solution. 

.. note::

   The current version of Doodl uses Tensorflow 1.15. Unfortunately, at this time, the `Tensorflow Object
   Detection <https://github.com/tensorflow/models/tree/master/research/object_detection>`_ project doesn't 
   support Tensorflow 2.x, so Doodl is limited by this restriction.


Assuming that you :doc:`already installed <installation>` Doodl including it's *tensorflow* backend, the 
following snippet of code shows how run inference on an image:

.. code-block:: python

    from doodl import Model
    
    model = Model()
    detections = model.inference("http://example.com/image.jpg")

    print(detections)

The :class:`~doodl.model.Model` class is the main interface provided by Doodl. Using the 
:meth:`~doodl.model.Model.inference` function you can run object detection on the provided image and return
the list of detected objects.


Remote object detection
-----------------------

Assuming you :doc:`installed the backend server <installation>`, you can run the object detection process 
remotely and put that responsibility on the server. 

The main advantage of using this method is to keep your local environment clear of the libraries needed by 
the process (specially Tensorflow 1.15.) Since you don't have to go through the installation process multiple 
times, it is very quick to get started on different projects if you can use a backend server to do all the work 
for you.

Remember that running the process remotely introduces latency that could end up being very costly, depending 
on your specific context. Keep this in mind when deploying your final solution.

Here is our previous example, this time running on a backend server:

.. code-block:: python

    from doodl import Model, Configuration

    configuration = Configuration()
    configuration.endpoint = "https://example.com/doodl/inference"
    
    model = Model(configuration)
    detections = model.inference("http://example.com/image.jpg")

    print(detections)

This time we are using the :class:`~doodl.configuration.Configuration` class to specify the endpoint where our 
backend is listening. Assuming that the library can communicate with the endpoint, the prediction results 
would be identical as before.

Check the section :ref:`configuring_backend_server` for more information about configuring the backend server and 
specifying different endpoints.


Caching predictions to enable experimentation
---------------------------------------------

By default, Doodl will not cache the inference results. You can change this behavior by using an instance
of the :class:`~doodl.configuration.Configuration` class when creating the model:

.. code-block:: python

    from doodl import Model, Configuration

    configuration = Configuration()
    configuration.cache = True
    
    model = Model(configuration)
    detections = model.inference("http://example.com/image.jpg")

    print(detections)

Now, calling the :meth:`~doodl.model.Model.inference` function repeatedly using the same image will make use
of the cached results instead of having to run the object detection process again.

Check the section :ref:`caching_predictions` for more information about how caching is managed by the library.


Specifying object detection model
---------------------------------

The backend implementation of Doodl comes with several models pre-installed. This makes it very simple to 
experiment with different implementations by merely specifying the name of the model you want the inference 
process to use. Here is an example snippet of code taking advantage of the *faster_rcnn_resnet50_coco* 
pre-trained model:

.. code-block:: python

    from doodl import Model, Configuration

    configuration = Configuration()
    configuration.endpoint = "https://example.com/doodl/inference"
    configuration.model = "faster_rcnn_resnet50_coco"
    
    model = Model(configuration)
    detections = model.inference("http://example.com/image.jpg")

    print(detections)

The pre-trained models do not come pre-packaged with the local implementation of Doodl, but you can still 
use any model you want with the library:

1. Download a compatible pre-trained model from the `Tensorflow detection model zoo <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md>`_ page.
2. Unzip the *.tar.gz* file in any location of your local environment. You will reference this directory from your code in the next step.
3. Set the :attr:`~doodl.configuration.Configuration.model_path` attribute to the unzipped directory that contains the *frozen_inference_graph.pb* file.
4. Set the :attr:`~doodl.configuration.Configuration.model` attribute to *frozen_inference_graph.pb*.

Here is how the code would look like assuming we'd like to use the *ssd_mobilenet_v1_coco_2018_01_28* pre-trained model:

.. code-block:: python

    from doodl import Model, Configuration

    configuration = Configuration()
    configuration.model_path = "/tmp/ssd_mobilenet_v1_coco_2018_01_28"
    configuration.model = "frozen_inference_graph.pb"
    
    model = Model(configuration)
    detections = model.inference("http://example.com/image.jpg")

    print(detections)

