# TensorFlow Docker Image to run Object Detection on SageMaker

## Building and pushing the docker image to ECR

First of all, configure AWS' Command Line Interface locally so you can access the AWS account. 
[Here is more information about configuring the CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).

Now, let's create a repository in AWS ECR to host our docker image. You can either do this using the web interface, or the command line
as follows. Make sure you specify a name for your repository where indicated by the `<repository-name>` argument:
```
# aws ecr create-repository --repository-name <repository-name>
```

In order to upload your docker image to ECR, you'll need to login first from your terminal. Running the following command
will output a `docker login ...` command that you can copy and paste in the terminal window to authenticate your session:
```
# aws ecr get-login --no-include-email
```

If you don't have your AWS account number around, you can run the following command to get it. You are going to need this 
account number to build and push the docker image to ECR:
```
# aws sts get-caller-identity --query Account
```

Building the docker image can be done running the following command:
```
# docker build -t <account-id>.dkr.ecr.us-east-1.amazonaws.com/<repository-name>:latest .
```

Notice that you have to replace `<account-id>` by your AWS account identifier and `<repository-name>` by the name of the 
repository that you created before.

After the image finishes building, you can push it up to ECR. The image is quite large so you can expect the operation to 
take a few minutes depending on the speed of your connection:
```
# docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/<repository-name>:latest
```

## Uploading relevant resources to S3

TBD

## Setting up a SageMaker training job

To train a model using the image you just uploaded, you need to create a training job using SageMaker's web interface. Here are the relevant
sections of the configuration of the training job that you need to update:

__Algorithm source:__ Here you'll select "Your own algorithm container in ECR". In the container path field you will specify your image 
`<account-id>.dkr.ecr.us-east-1.amazonaws.com/<repository-name>:latest`. (Make sure you replace the `<account-id>` and `<repository-name>`
placeholders.)

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
our docker image:

* `testing`: This channel will expose our data and configuration files to our docker image. Make sure to set the channel name property to 
`testing`, the input mode to `File`, the data source to `S3`, and the S3 location to the S3 location we created before.

* `checkpoint`: The second channel will expose our pre-trained network to our docker image. Set the channel name property to 
`checkpoint`, the input mode to `File`, the data source to `S3`, and the S3 location to the S3 folder that contains our pre-trainer network files.

__Output data configuration:__ When our model finishes training, SageMaker will upload the final resources to this location. Set this field to 
the S3 location where you want to store the output of the training process. 

At this point, and unless you want to tweak any of the other settings on the training job screen, you can create your training job. SageMaker will
initiate the training process and will inform you when it finished.

## Hosting your model on SageMaker
To get your model up and running on SageMaker, create a new Notebook instance using SageMaker's web interface and open Jupyter as soon as it's ready.
On that notebook, you can add the following code in a cell:

```
import boto3
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri

role = get_execution_role()

sagemaker = boto3.Session().client(service_name='sagemaker') 

# Before anything else, we need to set these variables appropriately.
#
# TRAINING_JOB_NAME should be the name of the training job that you
# used to train your model.
#
# DOCKER_IMAGE should point to the docker image that you created to
# train and serve your model.
TRAINING_JOB_NAME = 'nee-im-pets-7'
DOCKER_IMAGE = '116894939376.dkr.ecr.us-east-1.amazonaws.com/tfod:latest'

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

# Now, we need to create an Endpoint configuration. Make sure you 
# select an appropriate instance type to run inference on your model.
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
    print('Endpoint status: ', status)

if status != 'InService':
    print('Endpoint creation failed')

```

It will take some time to create and provision the instance running our docker image, but when is done, you should be ready
to test it with the code below.

First, let's define a simple function to visualize the predictions (detections) made by our model:

```
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def visualize_prediction(image_file, predictions, classes=[], threshold=0.6):
    image = mpimg.imread(image_file)
    plt.imshow(image)
    
    height = image.shape[0]
    width = image.shape[1]
    
    for prediction in predictions['prediction']:
        (class_id, confidence, x0, y0, x1, y1) = prediction
        if confidence >= threshold:
            cls_id = int(klass)
            xmin = int(x0 * width)
            ymin = int(y0 * height)
            xmax = int(x1 * width)
            ymax = int(y1 * height)
            
            rectangle = plt.Rectangle(
                (xmin, ymin), 
                xmax - xmin, 
                ymax - ymin, 
                fill=False, 
                edgecolor=(0, 0, 255), 
                linewidth=2.0)
            
            plt.gca().add_patch(rectangle)
            class_name = classes[cls_id]
            
            plt.gca().text(
                xmin, 
                ymin - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor=(0, 0, 255), alpha=0.5), fontsize=12, color='white')
    plt.show()
```

Then, we can download a sample image, and invoke our model endpoint to obtain the list of predictions back:

```
import json
import base64

# Let's download a sample image from the web and save it locally as test.jpg.
!wget -O test.jpg http://farm8.staticflickr.com/7198/6933992703_316b97d2cb_z.jpg
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
    classes=['Class A', 'Class B'], 
    threshold=0.2)
```

## Running your model on-premises
TBD
