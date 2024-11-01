from fastapi import status


class IException(Exception):
    def __init__(self, name: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR, message: str = None):
        self.name = name
        self.status_code = status_code
        self.detail = {"name": name, "status_code": status_code, "message": message}


class InternalServerException(IException):
    def __init__(self, message: str):
        super().__init__("Internal Server Error", status.HTTP_500_INTERNAL_SERVER_ERROR, message)


class ConflictException(IException):
    def __init__(self, name: str = "409 Conflict", message: str = "409 Conflict"):
        super().__init__(name, status.HTTP_409_CONFLICT, message)


class NotFoundException(IException):
    def __init__(self, name: str = " 404 Not Found", message: str = "404 Not Found"):
        super().__init__(name, status.HTTP_404_NOT_FOUND, message)
