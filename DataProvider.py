import random
import pandas as pd
from faker import Faker


LEGAL_FORM_VARIANTS = {
    "GmbH & Co. KG": ["GmbH&Co.KG", "GmbH & Co KG", "GmbH+Co.KG", "Ges.m.b.H & Co. KG"],
    "AG & Co. OHG":  ["AG&Co.OHG", "AG & Co OHG", "A.G. & Co. OHG"],
    "GmbH":          ["Ges.m.b.H", "Gesellschaft mbH", "GmbH.", "G.m.b.H."],
    "KGaA":          ["KG a.A.", "K.G.a.A."],
    "KG":            ["K.G.", "Kommanditges."],
    "OHG":           ["O.H.G.", "Offene HG"],
    "AG":            ["A.G.", "Aktienges.", "AG."],
}

STREET_ABBREVS = {
    "straße": ["str.", "strasse", "Str."],
    "gasse":  ["g.", "G."],
    "allee":  ["al.", "Al."],
    "platz":  ["pl.", "Pl."],
    "ring":   ["rng."],
    "weg":    ["w.", "Weg"],
}


def _typo(s: str, n: int = 1) -> str:
    chars = list(s)
    for _ in range(n):
        if len(chars) < 3:
            break
        i = random.randint(1, len(chars) - 2)
        op = random.choice(["swap", "drop", "dup"])
        if op == "swap":
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "drop":
            chars.pop(i)
        else:
            chars.insert(i, chars[i])
    return "".join(chars)


def _abbreviate_legal_form(name: str) -> str:
    # Längste Treffer zuerst, sonst matcht "GmbH" vor "GmbH & Co. KG"
    for form, variants in LEGAL_FORM_VARIANTS.items():
        if form in name:
            return name.replace(form, random.choice(variants), 1)
    return name


def _abbreviate_street(address: str) -> str:
    lower = address.lower()
    for word, variants in STREET_ABBREVS.items():
        if word in lower:
            idx = lower.find(word)
            return address[:idx] + random.choice(variants) + address[idx + len(word):]
    return address


def _noise_number(s: str) -> str:
    digits = [(i, c) for i, c in enumerate(s) if c.isdigit()]
    if len(digits) >= 2:
        i1, i2 = random.sample(range(len(digits)), 2)
        pos1, pos2 = digits[i1][0], digits[i2][0]
        lst = list(s)
        lst[pos1], lst[pos2] = lst[pos2], lst[pos1]
        s = "".join(lst)
    return s


class SAPMasterDataGenerator:
    def __init__(self, seed=42):
        self.fake = Faker("de_DE")
        Faker.seed(seed)
        random.seed(seed)
        self.products = [
            "OsciJet Fluidik-Oszillator", "Abfüllanlage Type 3",
            "Laborabzug Pro X", "Titration Unit 5000", "Sensorik-Modul X1",
            "Industrie-Zentrifuge", "Reinraum-Schleuse V2"
        ]

    def generate_base_data(self, n=1000) -> pd.DataFrame:
        data = []
        for i in range(n):
            record = (
                f"{self.fake.company()} | "
                f"{self.fake.street_address()}, {self.fake.city()} | "
                f"{random.choice(self.products)}"
            )
            data.append({"id": i, "text": record})
        return pd.DataFrame(data)

    def generate_noisy_duplicates(self, df_base: pd.DataFrame) -> pd.DataFrame:
        """
        Schwierigkeitsgrade (gewichtet):
          easy    20% — 1-2 Tippfehler
          medium  40% — Rechtsform + Straße abgekürzt, Ziffern vertauscht
          hard    25% — ein Feld weglassen + Tippfehler
          extreme 15% — Feldreihenfolge tauschen + alle Abkürzungen
        """
        records = []
        for _, row in df_base.iterrows():
            parts = [p.strip() for p in row["text"].split("|")]
            difficulty = random.choices(
                ["easy", "medium", "hard", "extreme"],
                weights=[0.20, 0.40, 0.25, 0.15],
            )[0]

            if difficulty == "easy":
                idx = random.randint(0, len(parts) - 1)
                parts[idx] = _typo(parts[idx], n=random.randint(1, 2))

            elif difficulty == "medium":
                parts[0] = _abbreviate_legal_form(parts[0])
                if len(parts) > 1:
                    parts[1] = _abbreviate_street(parts[1])
                    parts[1] = _noise_number(parts[1])
                if len(parts) > 2 and random.random() < 0.4:
                    parts[2] = parts[2].swapcase()

            elif difficulty == "hard":
                if len(parts) > 1:
                    drop_idx = random.randint(0, len(parts) - 1)
                    parts = [p for i, p in enumerate(parts) if i != drop_idx]
                idx = random.randint(0, len(parts) - 1)
                parts[idx] = _typo(parts[idx], n=2)

            elif difficulty == "extreme":
                parts[0] = _abbreviate_legal_form(parts[0])
                if len(parts) > 1:
                    parts[1] = _abbreviate_street(parts[1])
                random.shuffle(parts)

            records.append({
                "ground_truth_id": row["id"],
                "text": " | ".join(parts),
            })

        return pd.DataFrame(records)


data_generator = SAPMasterDataGenerator()