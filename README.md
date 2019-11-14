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
docker build -t <account-id>.dkr.ecr.us-east-1.amazonaws.com/<repository-name>:latest .
```

Notice that you have to replace `<account-id>` by your AWS account identifier and `<repository-name>` by the name of the 
repository that you created before.

After the image finishes building, you can push it up to ECR. The image is quite large so you can expect the operation to 
take a few minutes depending on the speed of your connection:
```
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/<repository-name>:latest
```

