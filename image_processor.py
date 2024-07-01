# Returns a torch tensor from a PIL Image.

import PIL
import PIL.Image
import torch

from clean_images import resize_image
from PIL import Image
from torchvision import transforms

def process_image_as_pil(image: PIL.Image, image_size:int = 224) -> torch.tensor:
    """
    Takes a PIL Image and returns a torch tensor corresponding to the image.
    `image` is the image to tensorify and `image_size` is the size to resize the image to before tensorifying.
    """
    
    # Resizes the image to size `image_size`.
    cleaned_img = resize_image(image_size, image)

    # Turns the image to a tensor.
    image_transformed = transforms.PILToTensor()

    #Unsqueeze the tensor.
    image_transformed_unsqueezed = image_transformed(cleaned_img).unsqueeze(0)

    # Return the result.
    return image_transformed_unsqueezed