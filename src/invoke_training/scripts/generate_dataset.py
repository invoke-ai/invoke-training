import json
import os


def generate_metadata(folder_path):
    """
    Generates a metadata.jsonl file from a folder of images.
    If a .txt file with the same name exists, its text is used in the metadata.
    """
    metadata_file = os.path.join(folder_path, "metadata.jsonl")

    with open(metadata_file, "w") as jsonl_file:
        for file in os.listdir(folder_path):
            if file.endswith(".png"):
                txt_file = os.path.splitext(file)[0] + ".txt"
                txt_path = os.path.join(folder_path, txt_file)
                image_path = os.path.join("train", file)

                # Extract text if .txt file exists
                text = ""
                if os.path.exists(txt_path):
                    with open(txt_path, "r") as f:
                        text = f.read().strip()

                # Write to metadata.jsonl
                metadata = {"file_name": image_path, "text": text}
                jsonl_file.write(json.dumps(metadata) + "\n")

def main():
    # User is prompted to enter the path to the image folder
    folder_path = input("Enter the path to your image folder: ")
    
    # Validate the folder path
    if not os.path.exists(folder_path):
        print("The folder does not exist. Please check the path.")
        return

    # Check if the folder contains any images
    if not any(file.endswith('.png') for file in os.listdir(folder_path)):
        print("The folder has no images. No metadata file was generated.")
        return

    generate_metadata(folder_path)
    print("Metadata file has been generated.")

if __name__ == "__main__":
    main()

