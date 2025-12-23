import os
import platform
import secrets
import stat
import logging
from pathlib import Path
from typing import Tuple, Union, Optional

try:
    import keyring  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    keyring = None  # fall back to file backend


class SecretManager:
    """Encapsulated secret storage and retrieval utility.

    Provides file-based and OS keyring backends with auto-selection and
    one-time migration on Windows/macOS when keyring is available.
    """

    # Class-level configuration
    APP_DIR_NAME = "exegol-mcp"
    SECRET_FILE_NAME = "bearer.secret"
    KEYRING_SERVICE = "exegol-mcp"
    KEYRING_USERNAME = "bearer"

    __bearer_secret: str | None = None

    # --------------- Path helpers ---------------
    @classmethod
    def __user_config_dir(cls) -> Path:
        system = platform.system().lower()
        if system == "windows":
            base = os.getenv("APPDATA") or os.path.expanduser("~\\AppData\\Roaming")
        elif system == "darwin":  # macOS
            base = os.path.expanduser("~/Library/Application Support")
        else:  # Linux/Unix
            base = os.getenv("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
        return Path(base) / cls.APP_DIR_NAME

    @classmethod
    def __get_secret_file_path(cls) -> Path:
        """Return the path where the bearer secret is stored on this OS."""
        return cls.__user_config_dir() / cls.SECRET_FILE_NAME

    # --------------- File backend helpers ---------------
    @classmethod
    def __ensure_dir_secure(cls, path: Path) -> None:
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            try:
                path.chmod(0o700)
            except Exception as e:
                logging.debug(f"Could not chmod config dir '{path}': {e}")

    @classmethod
    def __obfuscator(cls, secret: Union[str, bytes]) -> Union[bytes, str]:
        if type(secret) is str:
            data = secret.encode("utf-8")
        else:
            data = secret
        k = int.from_bytes(b"\x42")
        return bytes(c ^ k for c in data)

    @classmethod
    def __store_in_file(cls, path: Path, secret: str) -> None:
        cls.__ensure_dir_secure(path.parent)
        if os.name == "posix":
            # Create with 0o600 permissions atomically
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(cls.__obfuscator(secret))
        else:
            # On Windows, the file resides in the user's AppData. OS ACLs apply per-user.
            with open(path, "wb", encoding="utf-8") as f:
                f.write(cls.__obfuscator(secret))

    @classmethod
    def __harden_file_permissions(cls, path: Path) -> None:
        if os.name == "posix":
            try:
                st = path.stat()
                # If group/other have any permissions, restrict to 0o600
                if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                    path.chmod(0o600)
            except Exception as e:
                logging.debug(f"Could not harden permissions for '{path}': {e}")

    @classmethod
    def __load_password_from_file(cls, path: Path) -> str | None:
        try:
            if path.is_file():
                with open(path, "rb") as f:
                    secret = cls.__obfuscator(f.read()).decode("utf-8").strip()
                cls.__harden_file_permissions(path)
                if secret:
                    return secret
        except Exception as e:
            logging.warning(f"Failed to read bearer secret file '{path}': {e}")
        return None

    # --------------- Keyring helpers ---------------
    @classmethod
    def __is_keyring_available(cls) -> bool:
        return keyring is not None

    @classmethod
    def __get_keyring_backend_name(cls) -> str:
        if not cls.__is_keyring_available():
            return ""
        try:
            kr = keyring.get_keyring()
            # Some backends provide a readable name attribute, else use class name
            return getattr(kr, "name", kr.__class__.__name__)
        except Exception:
            return "unknown"

    @classmethod
    def __load_password_from_keyring(cls) -> str | None:
        if not cls.__is_keyring_available():
            return None
        try:
            return keyring.get_password(cls.KEYRING_SERVICE, cls.KEYRING_USERNAME)
        except Exception as e:  # pragma: no cover - system dependent
            logging.warning(f"Keyring get_password failed: {e}")
            return None

    @classmethod
    def __store_in_keyring(cls, secret: str) -> bool:
        if not cls.__is_keyring_available():
            return False
        try:
            keyring.set_password(cls.KEYRING_SERVICE, cls.KEYRING_USERNAME, secret)
            return True
        except Exception as e:  # pragma: no cover - system dependent
            logging.warning(f"Keyring set_password failed: {e}")
            return False

    # --------------- Public API ---------------
    @classmethod
    def load_bearer_secret(cls, backend: str = "auto") -> Tuple[str, bool, Optional[str]]:
        """
        Load the bearer secret from secure storage (file or OS keyring). Create a new
        cryptographically secure one on first run.

        Args:
            backend: 'auto' (default), 'keyring', or 'file'.

        Returns:
            (secret, created, location)

            location:
                - file path when using file backend
                - 'keyring:<backend name>' when using keyring
        """
        if cls.__bearer_secret is not None:
            return cls.__bearer_secret, False, "cached"

        system = platform.system().lower()
        use_keyring = False

        if backend == "keyring":
            if not cls.__is_keyring_available():
                raise RuntimeError("Keyring backend requested but 'keyring' package is not installed")
            use_keyring = True
        elif backend == "file":
            use_keyring = False
        else:  # auto
            # Prefer keyring on Windows/macOS when available
            if system in ("windows", "darwin") and cls.__is_keyring_available():
                use_keyring = True

        if use_keyring:
            # Try loading from keyring first
            secret = cls.__load_password_from_keyring()
            if secret:
                cls.__bearer_secret = secret
                return secret, False, f"keyring:{cls.__get_keyring_backend_name()}"

            # Create new secret and store in keyring
            new_secret = secrets.token_urlsafe(32)
            if cls.__store_in_keyring(new_secret):
                cls.__bearer_secret = new_secret
                return new_secret, True, f"keyring:{cls.__get_keyring_backend_name()}"
            else:
                logging.warning("Keyring not usable, falling back to file backend")
                # Fall back to file storage

        file_path = cls.__get_secret_file_path()

        # FILE BACKEND path (explicit file, or auto fallback)
        existing = cls.__load_password_from_file(file_path)
        if existing:
            cls.__bearer_secret = existing
            return existing, False, None

        # Generate and store new
        secret = secrets.token_urlsafe(32)
        try:
            cls.__store_in_file(file_path, secret)
        except Exception as e:
            logging.error(f"Failed to store bearer secret at '{file_path}': {e}")
            # Still return the secret to allow running, but warn that it's ephemeral
        cls.__bearer_secret = secret
        return secret, True, None
