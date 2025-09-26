import inspect
import os
import datetime

from backend.utils.config import config

class SmartLogger:
    def __init__(self, show_time: bool = True, show_level: bool = True):
        # print(f"Param loggers : {show_time} - {show_level}")
        self.show_time = show_time
        self.show_level = show_level
    
    def _get_caller_info(self) -> str:
        """Récupère les informations de l'appelant"""
        frame = inspect.currentframe().f_back.f_back.f_back
        # filename = os.path.basename(frame.f_code.co_filename)
        filename = os.path.relpath(frame.f_code.co_filename)
        function_name = frame.f_code.co_name
        return f"{filename} - {function_name}"
    
    def _format_message(self, level: str, message: str, overwrite: bool = False) -> str:
        """Formate le message avec les informations"""
        parts = []
        if self.show_time:
            parts.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}]")
        
        if self.show_level and not overwrite:
            parts.append(f"[{level} - {self._get_caller_info()}]")
        else:
            parts.append(f"[{self._get_caller_info()}]")
        parts.append(str(message))
        return " ".join(parts)
    
    def smartp(self, message: str):
        print(self._format_message("INFO", message, overwrite=True))
    def debug(self, message: str):
        print(self._format_message("DEBUG", message))
    def warning(self, message: str):
        print(self._format_message("⏹️ WARNING", message))
    def success(self, message: str):
        print(self._format_message("✅ SUCCESS", message))
    def error(self, message: str):
        print(self._format_message("❌ ERROR", message))

# Instance globale
smartlog = SmartLogger(show_time=config.logging.show_time, show_level=config.logging.show_level)
