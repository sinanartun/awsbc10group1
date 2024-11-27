import requests
import json
import time
# from transformers import AutoTokenizer, AutoModelForCausalLM

# Replace with your Telegram bot token
TELEGRAM_API_TOKEN = '7923576293:AAHCNs20pIo4SOwo-y0aJeGk2HLquwSTefo'
TELEGRAM_API_URL = f'https://api.telegram.org/bot{TELEGRAM_API_TOKEN}'

# Initialize LLaMA Model (make sure you've downloaded the model)
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with correct LLaMA model version
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# Define a function to send a message to a Telegram user
def send_telegram_message(chat_id, text):
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

# Define a function to generate a response using LLaMA API
def generate_response(prompt):
    url = 'http://localhost:8090/api/generate'
    payload = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json().get("response", "")

# Handle incoming updates from Telegram (long polling method)
def get_updates(offset=None):
    url = f"{TELEGRAM_API_URL}/getUpdates"
    params = {"offset": offset, "timeout": 100}
    response = requests.get(url, params=params)
    return response.json()

def load_local_image(image_path):
    """
    Loads an image from a local file path.
    """
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading local image: {e}")
        return None

# Main loop to check for messages and respond
def main():
    offset = None
                        response = generate_response(text)  # Generate response from LLaMA
                        send_telegram_message(chat_id, response)  # Send the response to Telegram bot
                        
                        # Update the offset to avoid reprocessing the same messages
                        offset = update["update_id"] + 1
        else:
            print("Error retrieving updates.")
        time.sleep(1)

if __name__ == "__main__":
    main()
