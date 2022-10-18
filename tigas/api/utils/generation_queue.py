from collections import deque
import yaml

# custom module
from .form import TIGAS_Form


# parse config file
with open('tigas.yaml', 'r') as f:
    config = yaml.safe_load(f)
    _MAX_SIZE = config['tigas']['generate']['tti']['max']
MAX_LEN = _MAX_SIZE

# FIFO queue for image generation model
TIGAS_QUEUE = deque(maxlen=MAX_LEN)

def append_to_queue(item:TIGAS_Form):
    TIGAS_QUEUE.append(item)

def pop_from_queue() -> TIGAS_Form:
    return TIGAS_QUEUE.popleft()

def get_queue_len() -> int:
    return len(TIGAS_QUEUE)

def get_index_of_item(item:TIGAS_Form) -> int:
    try:
        return TIGAS_QUEUE.index(item)
    except ValueError:
        return -1

def get_object_index_by_uuid(item:TIGAS_Form) -> int:
    for i, obj in enumerate(TIGAS_QUEUE):
        if obj.uuid == item.uuid:
            return i + 1
    return -1
