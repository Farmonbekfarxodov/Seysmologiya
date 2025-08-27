from django.contrib import admin
from .models import Station, Well, HydrogenSeismology


@admin.register(Station)
class StationAdmin(admin.ModelAdmin):
    list_display = ("api_code", "db_name")
    search_fields = ("api_code", "db_name")
    ordering = ("db_name",)


@admin.register(Well)
class WellAdmin(admin.ModelAdmin):
    list_display = ("db_name", "api_name", "station")
    search_fields = ("api_name", "db_name", "station__db_name")
    list_filter = ("station",)
    ordering = ("station__db_name", "db_name")


@admin.register(HydrogenSeismology)
class HydrogenSeismologyAdmin(admin.ModelAdmin):
    list_display = (
        "station_code", "well_code", "date",
        "he", "h2", "o2", "n2", "ch4", "co2"
    )
    search_fields = ("station_code", "well_code")
    list_filter = ("station_code", "well_code", "date")
    date_hierarchy = "date"
    ordering = ("-date",)
