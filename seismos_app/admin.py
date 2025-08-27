from django.contrib import admin
from .models import Skvajina, AllIzmereniya


@admin.register(Skvajina)
class SkvajinaAdmin(admin.ModelAdmin):
    list_display = ("naim", "Latitude", "Longitude")
    search_fields = ("naim",)
    list_filter = ("naim",)
    ordering = ("naim",)


@admin.register(AllIzmereniya)
class AllIzmereniyaAdmin(admin.ModelAdmin):
    list_display = ("stansiya", "skvajina", "izmereniya", "ssid_id")
    search_fields = ("stansiya", "skvajina", "izmereniya", "ssid_id")
    list_filter = ("stansiya", "skvajina")
    ordering = ("stansiya", "skvajina")
