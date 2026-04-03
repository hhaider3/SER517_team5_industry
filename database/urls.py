"""URL configuration for the Django migration project."""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("django-admin/", admin.site.urls),
    path("", include("ava_apps.main.urls")),
    path("auth/", include("ava_apps.accounts.urls")),
    path("admin/", include("ava_apps.admin_portal.urls")),
]

# Serve K-12 images in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)