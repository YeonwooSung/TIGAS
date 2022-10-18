class TIGAS_Form:
    def __init__(self, prompt:str, uuid:str, type:str='tti'):
        self.prompt = prompt
        self.uuid = uuid
        self.type = type
    
    def compare(self, other):
        assert type(other) == TIGAS_Form
        return self.prompt == other.prompt and self.type == other.type
    
    def compare_fully(self, other):
        return self.compare(other) and self.uuid == other.uuid
    
    def __eq__(self, __o: object) -> bool:
        return self.compare_fully(__o)
