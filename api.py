import os
import torch
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from infer import denoise_and_save # Assuming denoise_and_save is in infer.py
# You might need to adjust imports based on your project structure
# from main import load_model # If load_model is needed separately
# from data_preprocessing import preprocess_audio # If needed directly
# from utils import get_inverse_mel, get_mel_transform # If needed directly

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
OUTPUT_FOLDER = 'temp_outputs'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'} # Adjust as needed
MODEL_PATH = "denoising_autoencoder.pt" # Default model path, consider making this configurable

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure temporary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/denoise', methods=['POST'])
def denoise_file_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Add a suffix to avoid potential filename collisions in output
        base, ext = os.path.splitext(filename)
        output_filename = f"{base}_denoised{ext}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        try:
            file.save(input_path)

            # Determine device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            # Call the denoising function from infer.py
            # Make sure denoise_and_save handles model loading internally or load it here
            denoise_and_save(
                input_path=input_path,
                output_path=output_path,
                model_path=MODEL_PATH,
                device=device
                # Add any other necessary parameters your denoise_and_save expects
            )

            if not os.path.exists(output_path):
                 raise FileNotFoundError("Denoised file was not created.")

            # Send the denoised file back
            return send_file(output_path, as_attachment=True)

        except Exception as e:
            # Log the exception for debugging
            print(f"Error during denoising: {e}")
            # Potentially provide more specific error messages based on exception type
            return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500
        finally:
            # Clean up temporary files
            if os.path.exists(input_path):
                os.remove(input_path)
            # Keep output file until sent, then clean up if needed,
            # or implement a separate cleanup mechanism.
            # For simplicity here, we assume send_file handles it or cleanup happens later.
            # If send_file doesn't block until transfer complete, you might need `after_this_request`
            # @app.after_request
            # def cleanup(response):
            #    if os.path.exists(output_path):
            #       try:
            #           os.remove(output_path)
            #       except Exception as error:
            #           app.logger.error("Error removing or closing downloaded file handle", error)
            #    return response

    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    # Consider adding host='0.0.0.0' to make it accessible on your network
    # Use debug=False in a production environment
    app.run(debug=True, port=5000)
