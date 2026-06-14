"""
Firebase/Firestore configuration for GPT-5.x model settings.

This stores the API model configurations in Firestore so they're persistent
and can't be accidentally modified.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# Firebase config (from user's project)
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyCtpuLR-rdHoC461vrQ_gActkjHGRcBTas",
    "authDomain": "erosolar-llm.firebaseapp.com",
    "projectId": "erosolar-llm",
    "storageBucket": "erosolar-llm.firebasestorage.app",
    "messagingSenderId": "218680900515",
    "appId": "1:218680900515:web:797c1567fade8eca597816",
    "measurementId": "G-SVMR00S44H"
}

# GPT-5.x Model Configurations - LOCKED
# These are the ONLY models that should be used for enhancement
GPT5_MODELS = {
    "primary": "gpt-5.1-codex-mini",
    "backup": [
        "gpt-5.1-mini",
        "gpt-5-nano"
    ],
    "api_type": "responses",  # Use Responses API, NOT Chat Completions
    "fallback_to_chat": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]  # Only if GPT-5.x unavailable
}

# API Configuration
API_CONFIG = {
    "responses_api": {
        "endpoint": "responses.create",
        "params": {
            "input": "str",  # Prompt input
            "temperature": "float",
            "max_output_tokens": "int",
            "text": {"format": {"type": "json_object"}}  # For structured output
        }
    },
    "chat_completions_api": {
        "endpoint": "chat.completions.create",
        "params": {
            "messages": "list",
            "temperature": "float",
            "max_tokens": "int",  # NOT max_completion_tokens for older models
            "response_format": {"type": "json_object"}
        }
    }
}


@dataclass
class ModelConfig:
    """Model configuration for the training pipeline."""
    model_id: str
    api_type: str  # "responses" or "chat_completions"
    max_tokens: int = 2048
    temperature: float = 0.7
    is_gpt5: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**data)


def get_gpt5_models() -> List[str]:
    """Get list of GPT-5.x models in priority order."""
    return [GPT5_MODELS["primary"]] + GPT5_MODELS["backup"]


def get_model_config(model_id: str) -> ModelConfig:
    """Get configuration for a specific model."""
    is_gpt5 = model_id.startswith("gpt-5")
    api_type = "responses" if is_gpt5 else "chat_completions"

    return ModelConfig(
        model_id=model_id,
        api_type=api_type,
        is_gpt5=is_gpt5
    )


def save_to_local_cache(config: Dict[str, Any], filename: str = "model_config.json"):
    """Save configuration to local cache as backup."""
    cache_dir = Path(__file__).parent.parent / "cache" / "configs"
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(cache_dir / filename, "w") as f:
        json.dump(config, f, indent=2)


def load_from_local_cache(filename: str = "model_config.json") -> Optional[Dict[str, Any]]:
    """Load configuration from local cache."""
    cache_path = Path(__file__).parent.parent / "cache" / "configs" / filename
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return None


class FirestoreModelStore:
    """
    Store and retrieve model configurations from Firestore.

    This ensures model settings are persistent and can't be accidentally
    modified in the code.
    """

    def __init__(self):
        self._db = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of Firestore client."""
        if self._initialized:
            return

        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            # Initialize Firebase if not already done
            if not firebase_admin._apps:
                # Try to use default credentials or service account
                try:
                    cred = credentials.ApplicationDefault()
                    firebase_admin.initialize_app(cred, {
                        'projectId': FIREBASE_CONFIG['projectId']
                    })
                except Exception:
                    # Fallback: initialize without credentials (for local dev)
                    firebase_admin.initialize_app(options={
                        'projectId': FIREBASE_CONFIG['projectId']
                    })

            self._db = firestore.client()
            self._initialized = True
        except ImportError:
            print("firebase-admin not installed. Using local cache only.")
            self._initialized = False
        except Exception as e:
            print(f"Firebase initialization failed: {e}. Using local cache only.")
            self._initialized = False

    def save_model_config(self, config_name: str, config: Dict[str, Any]) -> bool:
        """Save model configuration to Firestore."""
        self._ensure_initialized()

        # Always save to local cache
        save_to_local_cache({config_name: config}, f"{config_name}.json")

        if not self._db:
            return False

        try:
            doc_ref = self._db.collection('model_configs').document(config_name)
            doc_ref.set(config)
            return True
        except Exception as e:
            print(f"Firestore save failed: {e}")
            return False

    def get_model_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Get model configuration from Firestore or local cache."""
        self._ensure_initialized()

        if self._db:
            try:
                doc_ref = self._db.collection('model_configs').document(config_name)
                doc = doc_ref.get()
                if doc.exists:
                    return doc.to_dict()
            except Exception as e:
                print(f"Firestore read failed: {e}")

        # Fallback to local cache
        cached = load_from_local_cache(f"{config_name}.json")
        if cached and config_name in cached:
            return cached[config_name]

        return None

    def save_gpt5_config(self) -> bool:
        """Save the GPT-5.x configuration to Firestore."""
        config = {
            "models": GPT5_MODELS,
            "api_config": API_CONFIG,
            "version": "1.0",
            "locked": True  # Flag to prevent modifications
        }
        return self.save_model_config("gpt5_config", config)

    def get_gpt5_config(self) -> Dict[str, Any]:
        """Get GPT-5.x configuration, falling back to hardcoded values."""
        config = self.get_model_config("gpt5_config")
        if config:
            return config

        # Return hardcoded defaults if nothing in store
        return {
            "models": GPT5_MODELS,
            "api_config": API_CONFIG,
            "version": "1.0",
            "locked": True
        }


# Singleton instance
_model_store: Optional[FirestoreModelStore] = None


def get_model_store() -> FirestoreModelStore:
    """Get or create the Firestore model store singleton."""
    global _model_store
    if _model_store is None:
        _model_store = FirestoreModelStore()
    return _model_store


def initialize_model_configs():
    """Initialize and save model configurations to Firestore."""
    store = get_model_store()
    store.save_gpt5_config()
    print("GPT-5.x model configurations saved to Firestore/cache")


if __name__ == "__main__":
    # Initialize configurations when run directly
    initialize_model_configs()

    # Print current config
    store = get_model_store()
    config = store.get_gpt5_config()
    print("\nCurrent GPT-5.x Configuration:")
    print(json.dumps(config, indent=2))
