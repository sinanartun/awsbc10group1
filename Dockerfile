# Use the AWS Lambda Python 3.9 base image
FROM public.ecr.aws/lambda/python:3.9

# Install required system tools and Python pip
RUN yum update -y && \
    yum install -y gcc python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Copy requirements.txt into the container
COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt

# Install Python dependencies from requirements.txt
RUN python3 -m pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Set the working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy Lambda function code
COPY lambda_function.py .

# Copy ONNX model
COPY model/model.onnx /opt/model/model.onnx

# Set environment variables
ENV S3_BUCKET_NAME=cardata33 \
    ONNX_MODEL_PATH=/opt/model/model.onnx

# Set the CMD for Lambda handler
CMD ["lambda_function.handler"]
