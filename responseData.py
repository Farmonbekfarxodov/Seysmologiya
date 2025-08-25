from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class LocalizedName:
    uz: str
    ru: str


@dataclass
class Station:
    id: int
    name: LocalizedName
    code: str
    latitude: str
    longitude: str


@dataclass
class WellType:
    id: int
    name: LocalizedName
    status: int
    created_at: str
    updated_at: str


@dataclass
class Well:
    id: int
    name: LocalizedName
    code: str
    latitude: str
    longitude: str
    grunt: str
    depth: str
    water_layer: str
    address: str
    station_code: str
    station: Station
    well_type_id: int
    well_type: WellType
    created_at: str
    updated_at: str


@dataclass
class Link:
    url: Optional[str]
    label: str
    active: bool


@dataclass
class Result:
    current_page: int
    data: List[Well]
    first_page_url: str
    from_: int
    last_page: int
    last_page_url: str
    links: List[Link]
    next_page_url: Optional[str]
    path: str
    per_page: int
    prev_page_url: Optional[str]
    to: int
    total: int


@dataclass
class Response:
    result: Result
    errors: Optional[str]
