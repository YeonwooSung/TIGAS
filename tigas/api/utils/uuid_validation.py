UUID_LEN = 36


def validate_uuid_encoding(uuid_str:str) -> bool:
    """
    Validate uuid encoding type.
    This method is implemented based on the assumption that the uuid is constructed with ascii characters.

    :param uuid_str: uuid string
    :return: True if uuid is constructed with ascii characters, False otherwise
    """
    try:
        uuid_str.encode('ascii')
    except UnicodeEncodeError:
        return False
    return True

def validate_uuid_len(uuid_str:str) -> bool:
    """
    Validate uuid length.
    This method is implemented based on the fact that the uuid is constructed with 32 characters and 4 dashes.
    """
    return len(uuid_str) == UUID_LEN

def validate_uuid(uuid_str:str) -> bool:
    '''
    Validate uuid string.
    This method is implemented based on the assumption that the uuid is constructed with 36 ascii characters (32 values and 4 dashes).

    :param uuid_str: uuid string
    :return: True if uuid is constructed with 36 ascii characters, False otherwise
    '''
    # Multi-levl validation
    #   1) check if encoding type is ascii
    #   2) check if length is 36
    return validate_uuid_encoding(uuid_str) and validate_uuid_len(uuid_str)
