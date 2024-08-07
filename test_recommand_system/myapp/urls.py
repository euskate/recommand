# myapp/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_view, name='index'),  # 루트 URL에 index_view를 연결
    path('chat/', views.chat_view, name='chat'),
    path('recommendation/', views.recommendation_view, name='recommendation'),
]