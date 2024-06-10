from clean_images import resize_image
from PIL import Image
from torchvision import transforms

def process_image(image_path:str, image_size:int = 64):
    with Image.open(image_path) as img:
        cleaned_img = resize_image(image_size, img)
    image_transformer = transforms.PILToTensor()
    return image_transformer(cleaned_img).unsqueeze(0)