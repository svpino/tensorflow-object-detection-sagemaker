Installation
============

There are two different ways to use Doodl: running the object detection process locally or
using a backend server. For more information and a comparison between these two approaches, 
take a look at the :doc:`introduction` section on this guide.


.. _local-installation: 

Running object detection locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can start by installing the library in your local environment. In this case, let's
include ``Tensorflow`` support:

.. code-block:: bash

   $ pip install doodl[tensorflow]

This will install Doodl, Tensorflow 1.15, and will set up your environment to run the object 
detection process locally. 

Assuming you don't get any errors during the installation, you can immediately start running predictions. 
Check out the :doc:`quickstart` guide for a list of examples.


.. _remote-installation:

Running object detection remotely
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Doodl comes with a Docker container that can be deployed as the backend server in charge 
of running the object detection process. This is especially useful when creating new applications
that can quickly take advantage of the service without having to set up the entire object
detection plumbing locally. 

To start the container you can use the public image published in Docker Hub and run it using the
following command:

.. code-block:: bash

   $ docker run --rm -p 8080:8080 svpino/doodl:0.1 serve

Starting the container with the ``serve`` command starts a ``gunicorn`` server listening for any 
HTTP POST requests to the ``/inference`` endpoint. Internally, this server listens on port ``8080``, 
so the above command maps that port to the same local port.

At this point you are ready to install the Python library, but this time you don't need to add ``Tensorflow``
support:

.. code-block:: bash

   $ pip install doodl

This will only install the layer that allows you to run the process remotely. Since, in 
this case, Tensorflow 1.15 is not needed, you can install and use a different version on 
the rest of your pipeline without having to worry about conflicting libraries.

Now check out the :doc:`quickstart` guide for a list of examples on how to use the library.