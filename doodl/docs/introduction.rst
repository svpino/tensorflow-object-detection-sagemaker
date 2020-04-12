Introduction to Doodl
=====================

While working on multiple projects using object detection, we found ourselves repeating 
over and over a lot of the necessary steps to take our idea from a design to a 
production-ready application. On top of that, working with images and videos is a 
time-consuming process, and we ended-up peppering our code with tricks and hacks to avoid
re-processing files during every experiment.

Doodl is the answer to those problems. It's a thin library designed to encapsulate the 
complexity of the `Tensorflow Object Detection 
API <https://github.com/tensorflow/models/tree/master/research/object_detection>`_, 
making the entire set up process dead simple, while offering additional functionality to 
cache results and run images and videos through the object detection process.


Object detection in your local environment
------------------------------------------

When developing a machine learning pipeline, the most common scenario is to run every 
component in your local environment. When :ref:`installing Doodl with Tensorflow support <local-installation>`, 
you'll get immediate access to the object detection API, and the entire process will run 
off your environment, taking advantage of the infrastructure you have available.
 
Running the process directly in your development environment means that you eliminate the
latency introduced by calling a remote endpoint, especially when running a lot of content. 

The downside is that you need to install all of the plumbing necessary to run the object 
detection process. Although Doodl makes the installation process very simple, it forces 
the installation of Tensorflow 1.15 in your environment. This may be a problem if you 
want to use Tensorflow 2.x in different components of your solution.


Object detection in a remote environment
----------------------------------------

For a team working on several projects at the same time, or for those wanting to take advantage 
of better hardware without needing to migrate their environment, Doodl comes with an 
:ref:`optional backend server <remote-installation>` —a Docker container— capable of isolating 
the object detection process away from your development environment.

Putting the process somewhere else comes with the added advantage of not needing Tensorflow 1.15 
in your environment. This is huge for developers that want to use the latest version of Tensorflow 
to develop other pieces of the pipeline. It also makes it very simple to get started on a new 
project without having to repeat a single line of code to get things running. 

Remember, however, that running the process remotely introduces latency that could add up very 
quickly. Keep this in mind when designing your solution.

Check out the :doc:`installation` instructions and then go to over the :doc:`quickstart` section
to get started.