import time
import torch
from PIL import Image

# custom modules
from api import utils
from api.v1.generate import TTI_QUEUE, IMG_DIR_PATH, LOG_DIR_PATH



def init_model():
    '''
    Initializes the model and returns it.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = utils.CustomTextToImageModel(utils.ModelConfig, device, from_pretrained=False)
    model.eval()
    return model


def generate_image(model, text):
    '''
    Generates an image from the given text.
    '''
    with torch.no_grad():
        image = model(text)
    # convert tensor to pillow image
    pil_image = convert_model_generate_img_to_pillow_img(image)
    return pil_image


def convert_model_generate_img_to_pillow_img(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images[0]


def inference_loop():
    model = init_model()
    interval = 1
    small_interval = 0.1

    inference_logger = utils.StableLogger(f'{LOG_DIR_PATH}model.log', name='inference_logger')

    queue = TTI_QUEUE
    while True:
        # check if something is in the queue
        if len(queue) > 0:
            # get the first item in the queue
            user_info = queue.popleft()
            uuid = user_info.uuid
            prompt = user_info.prompt

            # generate the image from text
            pil_image = generate_image(model, prompt)
            img_path = f'{IMG_DIR_PATH}{uuid}.png'
            # save the image
            pil_image.save(img_path)
            # save log
            inference_logger.log(f'user={uuid} - generated image for "{prompt}"')
        remaining_num = len(queue)
        print('remaining requests = ',remaining_num)
        currentInterval = small_interval if remaining_num > 0 else interval
        time.sleep(currentInterval)
