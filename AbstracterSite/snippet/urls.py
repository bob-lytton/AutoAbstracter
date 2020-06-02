from django.urls import path
from snippet import views

urlpatterns = [
    # path('snippet/', views.snippet_list),
    path('snippet/', views.snippet_list),
    path('snippet/<int:pk>/', views.snippet_detail),
]