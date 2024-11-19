import boto3
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import os
import json
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the S3 client
s3 = boto3.client('s3')

def preprocess_image(image_data, input_shape):
    """
    Preprocess the image to match the ONNX model input.
    Converts the image to RGB, resizes it, normalizes pixel values,
    and reshapes it to the model's input dimensions.
    """
    try:
        logger.info("Starting image preprocessing.")
        image = Image.open(image_data).convert("RGB")
        image = image.resize((input_shape[2], input_shape[3]))
        image_array = np.asarray(image).astype('float32') / 255.0  # Normalize to [0, 1]
        image_array = np.transpose(image_array, [2, 0, 1])  # Change to channels-first format
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        logger.info("Image preprocessing completed.")
        return image_array
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        raise

def download_image_from_s3(bucket_name, image_key):
    """
    Downloads an image from an S3 bucket.
    """
    try:
        logger.info(f"Downloading image from S3: bucket={bucket_name}, key={image_key}")
        image_data = io.BytesIO()
        s3.download_fileobj(bucket_name, image_key, image_data)
        image_data.seek(0)
        logger.info("Image downloaded successfully.")
        return image_data
    except Exception as e:
        logger.error(f"Error downloading image from S3: {e}")
        raise

def load_model(onnx_model_path):
    """
    Loads the ONNX model from the specified path.
    """
    try:
        logger.info(f"Loading ONNX model from path: {onnx_model_path}")
        session = ort.InferenceSession(onnx_model_path)
        logger.info("ONNX model loaded successfully.")
        return session
    except Exception as e:
        logger.error(f"Error loading ONNX model: {e}")
        raise

def adjust_input_shape(input_shape):
    """
    Adjusts the ONNX model input shape by replacing non-numeric dimensions with 1.
    """
    try:
        adjusted_shape = [
            int(dim) if isinstance(dim, (int, float)) else 1  # Replace non-numeric dimensions with 1
            for dim in input_shape
        ]
        logger.info(f"Adjusted input shape: {adjusted_shape}")
        return adjusted_shape
    except Exception as e:
        logger.error(f"Error adjusting input shape: {e}")
        raise

def run_inference(session, input_data):
    """
    Runs inference on the input data using the ONNX model.
    """
    try:
        logger.info("Running inference.")
        input_name = session.get_inputs()[0].name
        predictions = session.run(None, {input_name: input_data})
        logger.info("Inference completed.")
        return predictions
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

def handler(event, context):
    """
    AWS Lambda handler function to process an image from S3,
    run inference using an ONNX model, and return predictions.
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Get query string parameters
        bucket_name = os.environ.get('S3_BUCKET_NAME')
        image_key = event.get('queryStringParameters', {}).get('image_name')
        onnx_model_path = os.environ.get('ONNX_MODEL_PATH', '/opt/model/model.onnx')

        # Validate parameters
        if not bucket_name or not image_key:
            logger.error("Missing bucket name or image key.")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing bucket name or image key"})
            }

        # Download the image
        image_data = download_image_from_s3(bucket_name, image_key)

        # Load the ONNX model
        session = load_model(onnx_model_path)

        # Adjust input shape for preprocessing
        input_shape = adjust_input_shape(session.get_inputs()[0].shape)

        # Preprocess the image
        input_data = preprocess_image(image_data, input_shape)

        # Ensure input data is the correct type
        input_data = input_data.astype(np.float32)

        # Perform inference
        predictions = run_inference(session, input_data)

        # Return predictions as JSON
        return {
            "statusCode": 200,
            "body": json.dumps({
                "predictions": predictions[0].tolist()
            })
        }

    except Exception as e:
        logger.error(f"Handler encountered an error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
