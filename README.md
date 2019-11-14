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
TBD

## Running your model on-premises
TBD
