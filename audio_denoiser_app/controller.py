import shutil

def denoise_audio(input_path: str, output_path: str) -> str:
    shutil.copy(input_path, output_path)
    return output_path
