def validate_prompt(prompt:str):
    '''
    Validate that the given prompt is ascii only - no utf-8, no emoji, no special characters.

    @param prompt: the prompt to validate
    '''
    try:
        prompt.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False
