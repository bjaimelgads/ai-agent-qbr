"""Region and country heuristics for metric geo inference."""

from __future__ import annotations

import re


REGION_US = "US"
REGION_EMEA = "EMEA"


EMEA_COUNTRIES = {
    "Albania",
    "Algeria",
    "Andorra",
    "Angola",
    "Armenia",
    "Austria",
    "Azerbaijan",
    "Bahrain",
    "Belarus",
    "Belgium",
    "Benin",
    "Bosnia and Herzegovina",
    "Botswana",
    "Bulgaria",
    "Burkina Faso",
    "Burundi",
    "Cameroon",
    "Cape Verde",
    "Central African Republic",
    "Chad",
    "Comoros",
    "Congo",
    "Congo, Democratic Republic of the",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Djibouti",
    "Egypt",
    "Equatorial Guinea",
    "Eritrea",
    "Estonia",
    "Eswatini",
    "Ethiopia",
    "Finland",
    "France",
    "Gabon",
    "Gambia",
    "Georgia",
    "Germany",
    "Ghana",
    "Greece",
    "Guinea",
    "Guinea-Bissau",
    "Hungary",
    "Iceland",
    "Iran",
    "Iraq",
    "Ireland",
    "Israel",
    "Italy",
    "Ivory Coast",
    "Jordan",
    "Kazakhstan",
    "Kenya",
    "Kosovo",
    "Kuwait",
    "Kyrgyzstan",
    "Latvia",
    "Lebanon",
    "Lesotho",
    "Liberia",
    "Libya",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Madagascar",
    "Malawi",
    "Mali",
    "Malta",
    "Mauritania",
    "Mauritius",
    "Moldova",
    "Monaco",
    "Montenegro",
    "Morocco",
    "Mozambique",
    "Namibia",
    "Niger",
    "Nigeria",
    "North Macedonia",
    "Norway",
    "Oman",
    "Pakistan",
    "Palestine",
    "Poland",
    "Portugal",
    "Qatar",
    "Romania",
    "Russia",
    "Rwanda",
    "Saudi Arabia",
    "Senegal",
    "Serbia",
    "Sierra Leone",
    "Slovakia",
    "Slovenia",
    "Somalia",
    "South Africa",
    "South Sudan",
    "Spain",
    "Sudan",
    "Sweden",
    "Switzerland",
    "Syria",
    "Tanzania",
    "Togo",
    "Tunisia",
    "Turkey",
    "Uganda",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom",
    "UK",
    "UAE",
    "England",
    "Wales",
    "Scotland",
    "Northern Ireland",
    "Uzbekistan",
    "Yemen",
    "Zambia",
    "Zimbabwe",
}


US_SYNONYMS = {"United States", "United States of America", "USA", "U.S.", "US", "Domestic"}


def normalize_country_token(token: str) -> str:
    return re.sub(r"\s+", " ", token.strip())


def infer_country_from_text(text: str) -> str | None:
    if not text:
        return None
    lowered = text.lower()
    for country in sorted(EMEA_COUNTRIES, key=len, reverse=True):
        if country.lower() in lowered:
            if country == "UK":
                return "United Kingdom"
            if country == "UAE":
                return "United Arab Emirates"
            return country
    for country in sorted(US_SYNONYMS, key=len, reverse=True):
        if country.lower() in lowered:
            return "United States"
    return None


def infer_region_from_text(text: str) -> str | None:
    if not text:
        return None
    lowered = text.lower()
    if re.search(r"\bemea\b", lowered):
        return REGION_EMEA
    if re.search(r"\b(us|u\.s\.|usa|united states|domestic)\b", lowered):
        return REGION_US
    country = infer_country_from_text(text)
    if country == "United States":
        return REGION_US
    if country in EMEA_COUNTRIES:
        return REGION_EMEA
    return None
