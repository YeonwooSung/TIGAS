UUID_LEN = 36


def validate_uuid_encoding(uuid_str:str) -> bool:
    try:
        uuid_str.encode('ascii')
    except UnicodeEncodeError:
        return False
    return True

def validate_uuid_len(uuid_str:str) -> bool:
    return len(uuid_str) == UUID_LEN

def validate_uuid(uuid_str:str) -> bool:
    # Multi-levl validation
    #   1) check if encoding type is ascii
    #   2) check if length is 36
    return validate_uuid_encoding(uuid_str) and validate_uuid_len(uuid_str)
