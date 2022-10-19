import time
import torch
from PIL import Image

# custom modules
from api import utils
from api.v1.generate import IMG_DIR_PATH, LOG_DIR_PATH
from api.utils import pop_from_queue, get_queue_len



def init_model():
    '''
    Initializes the model and returns it.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = utils.CustomTextToImageModel(utils.ModelConfig, device, from_pretrained=False)
    model.eval()
    return model


def generate_image(model, text, text_inversion:bool=True, image:Image=None):
    '''
    Generates an image from the given text.
    '''
    with torch.no_grad():
        image = model(text, use_text_inversion=text_inversion, img=image)
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

    while True:
        # check if something is in the queue
        if get_queue_len() > 0:
            # get the first item in the queue
            user_info = pop_from_queue()
            uuid = user_info.uuid
            prompt = user_info.prompt
            task_type = user_info.type

            # Text Inversion
            if task_type == 'tti':
                # generate the image from text
                pil_image = generate_image(model, prompt)
                img_path = f'{IMG_DIR_PATH}{uuid}.png'
                # save the image
                pil_image.save(img_path)
                # save log
                inference_logger.log(f'user={uuid} - generated image for "{prompt}"')

            # Prompt guided image to image
            elif task_type == 'i2i':
                img = user_info.get_image()

                # img is None if either file path is not given or the file does not exist
                if img != None:
                    # generate image for img2img task
                    pil_image = generate_image(model, prompt, text_inversion=False, image=img)
                    img_path = f'{IMG_DIR_PATH}{uuid}.png'
                    # save the image
                    pil_image.save(img_path)
                    # save log
                    inference_logger.log(f'user={uuid} - image conversion with guidance prompt: "{prompt}"')
                else:
                    inference_logger.log(f'user={uuid} - image not found')

            # invalid task type
            else:
                inference_logger.log(f'unknown task type: {task_type}')

        remaining_num = get_queue_len()
        print('remaining requests = ',remaining_num)
        currentInterval = small_interval if remaining_num > 0 else interval
        time.sleep(currentInterval)
