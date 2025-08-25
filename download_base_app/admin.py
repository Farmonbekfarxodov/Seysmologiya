from django.contrib import admin
from .models import HydrogenSeismology
# Register your models here.

@admin.register(HydrogenSeismology)
class HydrogenSeismologyAdmin(admin.ModelAdmin):
    list_display = ["station_code", "well_code", "date", ]
    search_fields = ["station_code", "well_code", "date", ]
    list_filter = ["station_code", "well_code", "date", ]
