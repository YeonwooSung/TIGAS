class TTI_Form:
    def __init__(self, prompt:str, uuid:str):
        self.prompt = prompt
        self.uuid = uuid
    
    def compare(self, other):
        assert type(other) == TTI_Form
        return self.prompt == other.prompt
    
    def compare_fully(self, other):
        assert type(other) == TTI_Form
        return self.prompt == other.prompt and self.uuid == other.uuid
    
    def __eq__(self, __o: object) -> bool:
        return self.compare_fully(__o)
