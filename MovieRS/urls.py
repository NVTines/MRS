from django.urls import path
from . import views

urlpatterns = [
    path('userrate/', views.userrate, name='userrate'),
    path('', views.moviegridfw, name='moviegridfw'),
    path('detail/<int:idx>', views.moviesingle, name='moviesingle'),
]