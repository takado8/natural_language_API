class FunctionCallABS:
    def __init__(self, description):
        self.description = description

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

