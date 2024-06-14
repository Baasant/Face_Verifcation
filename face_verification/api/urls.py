from django.urls import path
from .views import verify_faces

urlpatterns = [
    path('verify_faces/', verify_faces),
]
