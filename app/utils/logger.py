from PySide6.QtCore import QObject, Signal

class UILogger(QObject):
    message = Signal(str)

    def __init__(self):
        super().__init__()

    def log(self, text: str):
        self.message.emit(text)
