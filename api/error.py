class APIError(Exception):
    status: int
    msg: str

    def __init__(self, msg: str, status: int):
        super().__init__()
        self.status = status
        self.msg = msg

    def __repr__(self):
        return self.msg

    def __str__(self):
        return self.msg
