from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GeoReference:
    ref_lat: float
    ref_lon: float
    earth_radius_m: float = 6_371_000.0

    @property
    def ref_rad(self) -> float:
        return math.radians(self.ref_lat)

    def geo_to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        if self.ref_lat == 0 and self.ref_lon == 0:
            return lat, lon # Actually x, y passed as lat, lon
        x = self.earth_radius_m * math.radians(lon - self.ref_lon) * math.cos(self.ref_rad)
        y = self.earth_radius_m * math.radians(lat - self.ref_lat)
        return x, y

    def xy_to_geo(self, x: float, y: float) -> tuple[float, float]:
        if self.ref_lat == 0 and self.ref_lon == 0:
            return x, y
        lon = self.ref_lon + math.degrees(x / (self.earth_radius_m * math.cos(self.ref_rad)))
        lat = self.ref_lat + math.degrees(y / self.earth_radius_m)
        return lat, lon


def haversine_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float, earth_radius_m: float) -> float:
    p1 = math.radians(lat_a)
    p2 = math.radians(lat_b)
    dp = math.radians(lat_b - lat_a)
    dl = math.radians(lon_b - lon_a)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * earth_radius_m * math.asin(math.sqrt(max(0.0, min(1.0, a))))
