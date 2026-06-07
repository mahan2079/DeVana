import secrets
import json
import os

class APIKeyManager:
    """Manages API keys for the DeVana REST API."""
    
    KEY_FILE = "api_keys.json"
    
    @staticmethod
    def generate_key():
        """Generate a new secure API key."""
        return secrets.token_urlsafe(32)
    
    @classmethod
    def save_key(cls, key_name: str, key_value: str):
        """Save an API key to the local store."""
        keys = cls.load_keys()
        keys[key_name] = key_value
        with open(cls.KEY_FILE, "w") as f:
            json.dump(keys, f, indent=4)
            
    @classmethod
    def load_keys(cls):
        """Load all saved API keys."""
        if not os.path.exists(cls.KEY_FILE):
            return {}
        try:
            with open(cls.KEY_FILE, "r") as f:
                return json.load(f)
        except:
            return {}

    @classmethod
    def validate_key(cls, key: str):
        """Check if a key is valid."""
        keys = cls.load_keys()
        return key in keys.values()
