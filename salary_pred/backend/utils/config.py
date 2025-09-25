import yaml
from typing import Dict, Any

class DotDict:
    """Permet l'accès par points aux dictionnaires imbriqués"""
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DotDict(value))
            else:
                setattr(self, key, value)

class Config:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
        self._setup_dynamic_attributes()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file) or dict()
        
    def _setup_dynamic_attributes(self):
        """Configure l'accès dynamique aux attributs"""
        for key, value in self._config.items():
            if isinstance(value, dict):
                # Crée un sous-objet pour les dictionnaires
                setattr(self, key, DotDict(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default=None) -> Any:
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

# Instance globale
config = Config()

