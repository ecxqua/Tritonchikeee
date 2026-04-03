from fastapi import Request


def get_id_service(request: Request):
    return request.app.state.id_service


def get_temp(request: Request):
    return request.app.state.temp