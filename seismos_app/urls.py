from django.urls import path
from .import views

urlpatterns = [
    path('',views.selection_view,name='selection'),
    path('parametrs/',views.parametrs_view,name='parametrs'),
    path('results/',views.results_view,name='results'),
]