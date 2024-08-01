from PIL import Image
from torchvision import transforms

def preprocess_frame(frame):
    # Convert numpy array of the frame to a PIL Image object
    frame = Image.fromarray(frame)
    
    # Define preprocessing steps: resize to 128x128 and convert to PyTorch tensor
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    # Apply preprocessing and add batch dimension
    return preprocess(frame).unsqueeze(0)