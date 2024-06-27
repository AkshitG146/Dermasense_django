from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    # path('handle-image/', views.handle_image, name='image_handler'),
]