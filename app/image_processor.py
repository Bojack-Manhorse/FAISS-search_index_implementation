import PIL
import PIL.Image

from clean_images import resize_image
from PIL import Image
from torchvision import transforms

def process_image_as_pil(image: PIL.Image, image_size:int = 224):
    cleaned_img = resize_image(image_size, image)
    image_transformer = transforms.PILToTensor()
    return image_transformer(cleaned_img).unsqueeze(0)