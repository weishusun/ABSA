"""Cleaning modules."""

from .base import BaseCleaner
from .domains.phone import PhoneCleaner
from .domains.car import CarCleaner
from .domains.laptop import LaptopCleaner
from .domains.beauty import BeautyCleaner

__all__ = [
    "BaseCleaner",
    "PhoneCleaner",
    "CarCleaner",
    "LaptopCleaner",
    "BeautyCleaner",
]
