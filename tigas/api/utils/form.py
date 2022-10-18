import os
from PIL import Image


class TIGAS_Form:
    def __init__(self, prompt:str, uuid:str, type:str='tti', img_path:str=None):
        self.prompt = prompt
        self.uuid = uuid
        self.type = type
        self.img_path = img_path
    
    def compare(self, other) -> bool:
        assert type(other) == TIGAS_Form
        return self.prompt == other.prompt and self.type == other.type
    
    def compare_fully(self, other) -> bool:
        return self.compare(other) and self.uuid == other.uuid
    
    def get_img_path(self) -> str:
        return self.img_path
    
    def get_image(self) -> Image:
        if self.img_path != None:
            if os.path.exists(self.img_path):
                return Image.open(self.img_path).convert('RGB')
            return None
        else:
            return None

    def __eq__(self, __o: object) -> bool:
        return self.compare_fully(__o)
