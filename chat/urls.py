from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_view, name='chat'),
    path('api/perguntar/', views.chat_api, name='chat_api'),
]
