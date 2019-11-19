# TensorFlow Docker Image to run Object Detection on SageMaker

## Building and pushing the docker image to ECR

First of all, configure AWS' Command Line Interface locally so you can access the AWS account. 
[Here is more information about configuring the CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).

Now, let's create a repository in AWS ECR to host our docker image. You can either do this using the web interface, or the command line
as follows. Make sure you specify a name for your repository where indicated by the `<repository-name>` argument:
```sh
aws ecr create-repository --repository-name <repository-name>
```

If you don't have your AWS account identifier around, you can run the following command to get it. You are going to need this 
account number to build and push the docker image to ECR:
```sh
aws sts get-caller-identity --query Account
```

In order to upload your docker image to ECR, you'll need to login first from your terminal. Running the following command
will get your session authenticated (make sure you replace `<account-id>` by your account identifier from the previous
step):
```sh
$(aws ecr get-login --no-include-email --registry-id <account-id>)
```

To build your docker image you'll use the following command:
```sh
docker build 
    -t <account-id>.dkr.ecr.us-east-1.amazonaws.com/<repository-name>:1.15.0-gpu 
    --build-arg ARCHITECTURE=1.15.0-gpu .
```

Notice that you have to replace `<account-id>` by your AWS account identifier and `<repository-name>` by the name of the 
repository that you created before. Also, the `ARCHITECTURE` build argument supports the specific tag of the base image
depending on which version you want to build. For example:
* To build an image from the latest version of TensorFlow with GPU support, set `ARCHITECTURE=1.15.0-gpu`.
* To build an image from the latest version of TensorFlow with CPU support, set `ARCHITECTURE=1.15.0`.

After the image finishes building, you can push it up to ECR. The image is quite large so you can expect the operation to 
take a few minutes depending on the speed of your connection:
```sh
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/<repository-name>:1.15.0-gpu
```

## Uploading relevant resources to S3

Now that your docker image is out of the way, you have to create and upload all the relevant resources to S3 so SageMaker can make them
available to your image. Assuming you want to train your model on SageMaker (instead of just serve a model that was already trained) here 
is what you need to upload:

* TFRecord files with your training and validation data. You can read more about TFRecords 
[here](https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecords_format_details). These files will contain your training and 
validation images plus annotations. 

* The pipeline configuration file. This file contains the configuration of the specific algorithm that you use for training. A template of
the appropriate file comes with each one of the pre-trained models offered on 
[this repository](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

* The pre-trained model checkpoint. If you are planning to use Transfer Learning to kickstart your training, you'll need to provide the 
pre-trained model checkpoint so the algorithm can start from there.

* The list of labels you are trying to detect. This is usually provided as a `label_map.pbtxt` file. 
[Here is an example](https://github.com/tensorflow/models/blob/master/research/object_detection/data/pet_label_map.pbtxt) illustrating the 
format of this file.

To upload all of these files, first create a bucket in S3 and here and follow these instructions:

1. The TFRecord files with your training and validation data could be named however you like. You could also upload multiple TFRecord files 
(following TensorFlow's convention of naming these files as yourfile.record-00000-00010, and so on and so forth.)

2. The pipeline configuration file should be uploaded with the name `pipeline.config`. If you wish to change this name, you'll have to modify
it in the `train` file that's part of the docker image. Before uploading this file to S3, make sure you follow the instructions [explained 
below](#modifying-pipelineconfig) around modifying this file.

3. Upload the entire pre-trained model that you get from [the model zoo page](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Unzip it first, and upload the entire folder. It can be named however you'd like.

4. The label map file should be uploaded as `label_map.pbtxt`. If you wish to change this name, you'll have to modify it in the `train` and
`serve` files that are part of the docker image.

At the end, the content of your S3 bucket will look something like this:

```
- faster_rcnn_resnet50_coco_2018_01_28/
- training.record
- validation.record
- pipeline.config
- label_map.pbtxt
```

## Modifying pipeline.config
TBD

## Setting up a SageMaker training job

To train a model using the image you just uploaded, you need to create a training job using SageMaker's web interface. Here are the relevant
sections of the configuration of the training job that you need to update:

__Algorithm source:__ Here you'll select "Your own algorithm container in ECR". In the container path field you will specify your image 
`<account-id>.dkr.ecr.us-east-1.amazonaws.com/<repository-name>:1.15.0-gpu`. (Make sure you replace the `<account-id>` and `<repository-name>`
placeholders. Also, make sure you specify the proper version of your image.)

__Resource configuration:__ Select an appropriate instance type for your training needs (make sure it's Accelerated Computing so it enables
GPU training,) and enough GBs of additional storage volume to your instance. Failing to supply enough storage space will cause the training 
job to fail because it won't be able to download the necessary files.

__Hyperparameters:__ The following list of hyperparameters are supported by the training script embedded in your docker image:

* `num_steps`: The number of training steps that the Object Detection algorithm will use to train your model. If not specified, this value
defaults to `100`.

* `quantize`: Whether you want to also generate a TensorFlow Lite (TFLite) model that can be run on mobile devices. If not specified, this value 
defaults to `False`.

* `image_size`: The size that images will be resized to if we generating a TFLite model (both width and height will be set to this size.) This 
parameter is ignored if `quantize` is `False`. If not specified, this value defaults to `300`.

* `inference_type`: The type of inference that will be used if we generating a TFLite model. This should be one of `QUANTIZED_UINT8` or `FLOAT` 
values. This parameter is ignored if `quantize` is `False`. If not specified, this value defaults to `FLOAT`.

__Input data configuration:__ We want to create a couple of channels under this section to allow SageMaker to expose the necessary resources to
our docker image (it will do so by "mounting" a volume in our docker image so we can access the files directly from there):

* `training`: This channel will expose our data and configuration files to our docker image. Make sure to set the channel name property to 
`training`, the input mode to `File`, the data source to `S3`, and the S3 location to the S3 bucket we created before.

* `checkpoint`: The second channel will expose our pre-trained network to our docker image. Set the channel name property to 
`checkpoint`, the input mode to `File`, the data source to `S3`, and the S3 location to the S3 folder that contains our pre-trainer network files 
(this would be pointing to the `faster_rcnn_resnet50_coco_2018_01_28/` folder in our example above.)

__Output data configuration:__ When our model finishes training, SageMaker will upload your model results to this location. Set this field to 
the S3 location where you want to store the output of the training process. 

At this point, and unless you want to tweak any of the other settings on the training job screen, you can create your training job. SageMaker will
initiate the training process and will inform you when it finished.

## Hosting your model on SageMaker
To get your model up and running on SageMaker, create a new Notebook instance using SageMaker's web interface and open Jupyter as soon as it's ready.
On that notebook, you can add the following code in a cell:

```python
import boto3
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri

role = get_execution_role()

sagemaker = boto3.Session().client(service_name='sagemaker') 

# Before anything else, we need to set these variables appropriately.
#
# TRAINING_JOB_NAME should be the name of the training job that we
# used to train the model.
#
# DOCKER_IMAGE should point to the docker image that trains and serves
# the model.
TRAINING_JOB_NAME = <SAGEMAKER_TRAINING_JOB_NAME>
DOCKER_IMAGE = <DOCKER_IMAGE>
model_name = TRAINING_JOB_NAME

training_job = sagemaker.describe_training_job(TrainingJobName=TRAINING_JOB_NAME)

# Let's now create the model definition in SageMaker. 
response = sagemaker.create_model(
    ModelName = model_name,
    ExecutionRoleArn = role,
    PrimaryContainer = {
        'Image': DOCKER_IMAGE,
        'ModelDataUrl': training_job['ModelArtifacts']['S3ModelArtifacts'],
})

print('Model:', response['ModelArn'])

# Now, we need to create an Endpoint configuration. Make sure to select
# an appropriate instance type to run inference on the model.
endpoint_configuration_name = TRAINING_JOB_NAME + '-endpoint-configuration'

response = sagemaker.create_endpoint_config(
    EndpointConfigName = endpoint_configuration_name,
    ProductionVariants=[{
        'InstanceType':'ml.m4.xlarge',
        'InitialInstanceCount':1,
        'ModelName': model_name,
        'VariantName':'AllTraffic'}])

print('Endpoint configuration:', response['EndpointConfigArn'])

# Finally, we can create the Endpoint that will serve our model. This is going to 
# provision the appropriate instance and deploy our docker image so it can start 
# serving. This will take a while.
endpoint_name = TRAINING_JOB_NAME + '-endpoint'

response = sagemaker.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_configuration_name)

print('Creating endpoint', response['EndpointArn'])

status = 'Creating'
while status == 'Creating':
    # Let's wait until the status of the endpoint changes
    sagemaker.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)
    response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
    status = response['EndpointStatus']
    print('Endpoint status:', status)

if status != 'InService':
    print('Endpoint creation failed')
```

It will take some time to create and provision the instance running our docker image, but when is done, you should be ready
to test it with the code below.

First, let's define a simple function to visualize the predictions (detections) made by our model:

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def visualize_predictions(image_file, predictions, classes=[], threshold=0.6):
    image = mpimg.imread(image_file)
    plt.figure(figsize = (20,20))
    plt.imshow(image)
    
    height = image.shape[0]
    width = image.shape[1]
    
    for prediction in predictions['prediction']:
        (class_id, confidence, x0, y0, x1, y1) = prediction
        if confidence >= threshold:
            # Our model uses base 1 to represent the different classes, so
            # we need to convert it to base 0 for our purposes.
            class_id = int(class_id) - 1
            xmin = int(x0 * width)
            ymin = int(y0 * height)
            xmax = int(x1 * width)
            ymax = int(y1 * height)
            
            rectangle = plt.Rectangle(
                (xmin, ymin), 
                xmax - xmin, 
                ymax - ymin, 
                fill=False, 
                edgecolor='r', 
                linewidth=2.0)
            
            plt.gca().add_patch(rectangle)
            class_name = classes[class_id]
            
            plt.gca().text(
                xmin, 
                ymin - 2,
                '{:s} {:.3f}'.format(class_name, confidence),
                bbox=dict(facecolor='r', alpha=0.5), fontsize=12, color='white')
    plt.show()
```

Then, we can download a sample image, and invoke our model endpoint to obtain the list of predictions back:

```python
import json
import base64

# Let's download a sample image from the web and save it locally as test.jpg.
!wget -O test.jpg <IMAGE_URL>
image_file = 'test.jpg'

with open(image_file, "rb") as image:
    encoded_string = base64.b64encode(image.read())
    
# Now we can invoke our endpoint providing the image as a base64 string.
sagemaker-runtime = boto3.client('sagemaker-runtime')

response = sagemaker-runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=json.dumps({
        "image": encoded_string.decode('utf-8')
    }),
    ContentType='application/json'
)

# If everything works as expected, the response will contain a Body property
# containing the list of predictions inside.
predictions = json.loads(response['Body'].read().decode("utf-8"))

visualize_predictions(
    image_file=image_file, 
    predictions=predictions, 
    classes=['Class 1', 'Class 2'], 
    threshold=0.85)
```

## Running your model locally
You can run inference using your trained model locally by running the docker image on your computer. This is also
useful when you want to deploy your model on-premises and don't want to rely on AWS to use it.

To do this, first you'll need to download your trained model from S3. SageMaker saved your trained model in 
the S3 location that you specified when configuring your training job. Inside the folder that you specified, there's 
a `model.tar.gz` file that you will need to download and untar locally. You'll mount this folder to the docker image 
so it can use your model to run inference.

You can run the docker image that you built before, or you can build it again using a different tag to run it locally:

```sh
docker build 
    -t tensorflow-object-detection:1.15.0-cpu 
    --build-arg ARCHITECTURE=1.15.0 .
```

Starting the docker image with the `serve` command will start a gunicorn server that will be listening for any HTTP 
`POST` requests to the `/invocations` location. This server will be listening on port `8080`, so we need to make sure
to map that port to a local port.

Finally, when running locally, you can specify the timeout (`MODEL_SERVER_TIMEOUT`) and the number of workers (`MODEL_SERVER_WORKERS`)
that gunicorn will use through environment variables.

Here is an example command that will run the docker image and will make it listen to port 8080 locally:

```sh
docker run 
    -p 8080:8080 
    -v <local-path-to-model-folder>:/opt/ml/model 
    -e MODEL_SERVER_WORKERS=1
    --name "tensorflow-object-detection"
    tensorflow-object-detection:1.15.0-cpu serve
```

After having the docker image running, you can use the following script to run inference on an image (the script depends
on the `requests` library that you can install running `pip install requests`):

```python
import base64
import requests

if __name__ == '__main__':
    image_file = <LOCAL_IMAGE_FILE>

    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read())

    body = {
        "image": encoded_string.decode('utf-8')
    }

    response = requests.post('http://127.0.0.1:8080/invocations', json=body)

    print(response.json())
```