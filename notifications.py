"""Application-scoped, non-modal operation feedback."""
from PyQt6.QtCore import QObject, pyqtSignal


class MessageBus(QObject):
    posted = pyqtSignal(str, str, str)

    def post(self, level, title, message):
        self.posted.emit(level, str(title), str(message))


bus = MessageBus()


class Notice:
    @staticmethod
    def information(parent, title, message, *args, **kwargs):
        bus.post('成功', title, message)

    @staticmethod
    def warning(parent, title, message, *args, **kwargs):
        bus.post('提醒', title, message)

    @staticmethod
    def critical(parent, title, message, *args, **kwargs):
        bus.post('错误', title, message)
