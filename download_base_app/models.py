# data_importer/models.py

from django.db import models

class Station(models.Model):
    api_code = models.CharField(max_length=50, unique=True, help_text="API'dagi stansiya kodi (masalan, 'SMRM')")
    db_name = models.CharField(max_length=255, help_text="Bazada saqlanadigan to'liq nomi (masalan, 'Namangan KPS')")

    def __str__(self):
        return self.db_name

class Well(models.Model):
    station = models.ForeignKey(Station, on_delete=models.CASCADE, related_name='wells')
    api_name = models.CharField(max_length=255, help_text="API'dagi quduq nomi (masalan, 'Jumabozo'r 1')")
    db_name = models.CharField(max_length=255, help_text="Bazada saqlanadigan quduq nomi (masalan, 'Jumabozor 1')")

    def __str__(self):
        return f"{self.station.db_name} - {self.db_name}"

class HydrogenSeismology(models.Model):
    station_code = models.CharField(max_length=255, null=True, blank=True)
    well_code = models.CharField(max_length=255)
    date = models.DateTimeField(null=True, blank=True)
    he = models.FloatField(null=True, blank=True)
    h2 = models.FloatField(null=True, blank=True)
    o2 = models.FloatField(null=True, blank=True)
    n2 = models.FloatField(null=True, blank=True)
    ch4 = models.FloatField(null=True, blank=True)
    co2 = models.FloatField(null=True, blank=True)
    c2h6 = models.FloatField(null=True, blank=True)
    ph = models.FloatField(null=True, blank=True)
    eh = models.FloatField(null=True, blank=True)
    hco3 = models.FloatField(null=True, blank=True)
    ci2 = models.FloatField(null=True, blank=True)
    sio2 = models.FloatField(null=True, blank=True)
    f = models.FloatField(null=True, blank=True)
    i = models.FloatField(null=True, blank=True)
    b2o3 = models.FloatField(null=True, blank=True)
    dis_rn = models.FloatField(null=True, blank=True)
    nep_rn = models.FloatField(null=True, blank=True)
    he2 = models.FloatField(null=True, blank=True)
    t0 = models.FloatField(null=True, blank=True)
    q = models.FloatField(null=True, blank=True)
    p = models.FloatField(null=True, blank=True)
    eocc = models.FloatField(null=True, blank=True)
    nep_t0 = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.station_code} - {self.well_code} ({self.date})"