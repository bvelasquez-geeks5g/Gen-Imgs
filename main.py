import os
import uuid
import time
import logging
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from datetime import datetime
from google.cloud import storage
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration constants
LEONARDO_API_KEY = os.getenv('LEONARDO_API_KEY')
GCP_BUCKET_NAME = os.getenv('GCP_BUCKET_NAME')
GCP_FOLDER_NAME = os.getenv('GCP_FOLDER_NAME')
LOCAL_SAVE_PATH = os.getenv('LOCAL_SAVE_PATH', 'generated_images')
MAX_RETRY_ATTEMPTS = 3
MAX_GENERATION_ATTEMPTS = 10
GENERATION_WAIT_TIME = 10  # seconds between status checks
MODEL_ID = '458ecfff-f76c-402c-8b85-f09f6fb198de' #Deliberate 1.1

class LeonardoImageGenerator:
    def __init__(self):
        """
        Initialize the Leonardo AI image generation client
        with robust session and retry mechanism
        """
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=MAX_RETRY_ATTEMPTS,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)

        # Set up standard headers
        self.session.headers.update({
            "Authorization": f"Bearer {LEONARDO_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

        # Initialize Google Cloud Storage client
        try:
            self.storage_client = storage.Client.from_service_account_json(
                os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            )
            self.bucket = self.storage_client.bucket(GCP_BUCKET_NAME)
        except Exception as e:
            logger.error(f"Google Cloud Storage initialization failed: {e}")
            raise

        # Ensure local save path exists
        os.makedirs(LOCAL_SAVE_PATH, exist_ok=True)

    def generate_image(self, prompt, width=896, height=1192):
        """
        Generate an image with comprehensive error handling
        """
        try:
            # Construct enhanced prompt
            full_prompt = (
                f"{prompt}, "
                "ultra-photorealistic, cinematic lighting, "
                "high detail, natural textures, "
                "no distortions, no artifacts, no woman, no person, no people, no men, no man"
            )

            payload = {
                "prompt": full_prompt,
                "negative_prompt": (
                    "blurry, distorted, unnatural anatomy, "
                    "low resolution, artifacts, unrealistic, "
                    "low quality, text, watermark, distorted "
                    ""
                ),
                "height": height,
                "width": width,
                "num_images": 1,
                "modelId": MODEL_ID
            }

            # Initial generation request
            response = self.session.post(
                "https://cloud.leonardo.ai/api/rest/v1/generations",
                json=payload
            )
            response.raise_for_status()

            generation_id = response.json().get("sdGenerationJob", {}).get("generationId")
            if not generation_id:
                logger.warning("No generation ID received")
                return None

            # Poll for image generation status
            return self._poll_generation_status(generation_id)

        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None

    def _poll_generation_status(self, generation_id):
        """
        Poll Leonardo AI for image generation status
        """
        for attempt in range(MAX_GENERATION_ATTEMPTS):
            try:
                time.sleep(GENERATION_WAIT_TIME)

                status_url = f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}"
                response = self.session.get(status_url)
                response.raise_for_status()

                result_data = response.json().get("generations_by_pk", {})
                if result_data.get("status") == "COMPLETE":
                    image_data = result_data.get("generated_images", [{}])[0]

                    # NSFW check
                    if image_data.get('nsfw', False):
                        logger.warning("Generated image is NSFW")
                        return None

                    # Download image
                    image_url = image_data.get('url')
                    if not image_url:
                        logger.warning("No image URL found")
                        return None

                    image_response = self.session.get(image_url)
                    image_response.raise_for_status()

                    result_data['image_bytes'] = image_response.content
                    return result_data

            except requests.RequestException as e:
                logger.error(f"Status polling error (Attempt {attempt + 1}): {e}")

        logger.error("Maximum generation attempts reached")
        return None

    def upload_to_gcs(self, image_data):
        """
        Upload image to Google Cloud Storage with robust error handling
        """
        try:
            image_bytes = image_data.get('image_bytes')
            if not image_bytes:
                raise ValueError("No image bytes found")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_id = f"{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
            local_path = os.path.join(LOCAL_SAVE_PATH, image_id)

            # Save locally
            # with open(local_path, "wb") as img_file:
            #     img_file.write(image_bytes)

            # Upload to GCS
            blob = self.bucket.blob(f"{GCP_FOLDER_NAME}{image_id}")
            blob.upload_from_string(image_bytes, content_type="image/jpeg")
            blob.make_public()

            return blob.public_url

        except Exception as e:
            logger.error(f"GCS upload failed: {e}")
            return None


# Flask App Setup
app = Flask(__name__)
image_generator = LeonardoImageGenerator()


@app.route('/generate-image', methods=['POST'])
def generate_image_endpoint():
    """
    Image generation endpoint with comprehensive error handling
    """
    try:
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({'error': 'A prompt is required'}), 400

        prompt = data['prompt']
        width = min(data.get('width', 896), 900)
        height = min(data.get('height', 1192), 1536)

        generated_image = image_generator.generate_image(prompt, width, height)
        if not generated_image:
            return jsonify({'error': 'Image generation failed'}), 500

        image_url = image_generator.upload_to_gcs(generated_image)
        if not image_url:
            return jsonify({'error': 'Image upload failed'}), 500

        return jsonify({
            'success': True,
            'prompt': prompt,
            'image_url': image_url
        })

    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        return jsonify({'error': 'Unexpected server error'}), 500


if __name__ == '__main__':
    app.run(
        debug=False,
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000))
    )