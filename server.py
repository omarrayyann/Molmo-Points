import socket
import threading
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
import io

# Load the processor
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# Server settings
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 8080       # Arbitrary non-privileged port

def handle_client(conn, addr):
    print(f"Connected by {addr}")
    try:
        # Receive the size of the incoming data (4 bytes for the size)
        data_size_bytes = conn.recv(4)
        if not data_size_bytes:
            print("No data size received, closing connection.")
            return
        data_size = int.from_bytes(data_size_bytes, 'big')
        print(f"Expecting {data_size} bytes of data.")
        
        # Receive the actual data
        data = b''
        while len(data) < data_size:
            packet = conn.recv(4096)
            if not packet:
                print("Connection closed unexpectedly.")
                break
            data += packet
            print(f"Received {len(packet)} bytes, total received {len(data)}/{data_size} bytes.")
        
        if len(data) != data_size:
            print(f"Received incomplete data: {len(data)}/{data_size}")
            return

        # Separate the data into image and prompt
        separator = b'--SEPARATOR--'
        if separator not in data:
            conn.sendall(b"Error: Data format incorrect")
            print("Data separator not found, closing connection.")
            return
        image_data, prompt_data = data.split(separator)
        print("Data successfully split into image and prompt.")

        # Load the image from PNG bytes as a PIL image
        image = Image.open(io.BytesIO(image_data))
        print("Image successfully loaded from PNG data.")

        # Decode the prompt
        prompt = prompt_data.decode('utf-8')
        print(f"Prompt received: {prompt}")

        # Process the image and text exactly as requested
        inputs = processor.process(
            images=[image],  # Use the PIL image directly
            text=prompt
        )
        print("Input successfully processed by model processor.")

        # Move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        
        # Generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings=["<|endoftext|>"]),
            tokenizer=processor.tokenizer
        )
        print("Model generation completed.")

        # Only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"Generated text: {generated_text}")

        # Send back the generated text
        conn.sendall(generated_text.encode('utf-8'))
    except Exception as e:
        error_message = f"Error: {str(e)}"
        conn.sendall(error_message.encode('utf-8'))
        print(f"Exception: {error_message}")
    finally:
        conn.close()

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr))
            client_thread.start()

if __name__ == "__main__":
    start_server()

