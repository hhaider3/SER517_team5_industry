"""Django settings for the Adaptive Virtual Assistant migration."""
from pathlib import Path
import os
from dotenv import load_dotenv
from django.contrib.messages import constants as message_constants

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from the existing .env file if present.
env_file = BASE_DIR / ".env"
if env_file.exists():
    load_dotenv(env_file)


def _parse_int(value: str, fallback: int) -> int:
    """Internal helper to parse int."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return fallback


SECRET_KEY = os.getenv("DJANGO_SECRET_KEY") or os.getenv("SECRET_KEY") or "unsafe-dev-key"
DEBUG = os.getenv("DJANGO_DEBUG", "1") == "1"

allowed_hosts = os.getenv("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1")
ALLOWED_HOSTS = [host.strip() for host in allowed_hosts.split(",") if host.strip()]

csrf_trusted = os.getenv("DJANGO_CSRF_TRUSTED_ORIGINS", "")
CSRF_TRUSTED_ORIGINS = [host.strip() for host in csrf_trusted.split(",") if host.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "ava_apps.main",
    "ava_apps.accounts",
    "ava_apps.admin_portal",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

AUTHENTICATION_BACKENDS = [
    "ava_apps.accounts.auth_backends.DatabaseServiceBackend",
    "django.contrib.auth.backends.ModelBackend",
]

LOGIN_URL = "/auth/login"
LOGIN_REDIRECT_URL = "/learning-goal"
LOGOUT_REDIRECT_URL = "/"

ROOT_URLCONF = "ava_django.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates_django"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ]
        },
    }
]

WSGI_APPLICATION = "ava_django.wsgi.application"
ASGI_APPLICATION = "ava_django.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": os.getenv("DB_NAME", "adaptive_assistant"),
        "USER": os.getenv("DB_USER", "adaptive_user"),
        "PASSWORD": os.getenv("DB_PASSWORD", ""),
        "HOST": os.getenv("DB_HOST", "localhost"),
        "PORT": str(_parse_int(os.getenv("DB_PORT"), 3306)),
        "OPTIONS": {"charset": "utf8mb4"},
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = os.getenv("DJANGO_TIME_ZONE", "UTC")
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]

# --- K-12 Image media serving ---
MEDIA_URL = "/media/"
_SER517_DB_DIR = Path(os.getenv(
    "SER517_DB_DIR",
    str(BASE_DIR.parent / "SER517_team5_industry" / "database"),
))
MEDIA_ROOT = str(_SER517_DB_DIR)
IMAGE_DB_PATH = os.getenv("IMAGE_DB_PATH", str(_SER517_DB_DIR / "image_db"))

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

MESSAGE_TAGS = {
    message_constants.ERROR: "danger",
}
