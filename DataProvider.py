import random
import pandas as pd
from faker import Faker
from typing import List, Dict, Tuple, Final

class SAPMasterDataGenerator:
    """
    High-performance generator for synthetic SAP master data and noisy duplicates
    using Matryoshka-compatible noise patterns.
    """

    LEGAL_FORM_VARIANTS: Final[Dict[str, List[str]]] = {
        "GmbH & Co. KG": ["GmbH&Co.KG", "GmbH & Co KG", "GmbH+Co.KG", "Ges.m.b.H & Co. KG"],
        "AG & Co. OHG":  ["AG&Co.OHG", "AG & Co OHG", "A.G. & Co. OHG"],
        "GmbH":          ["Ges.m.b.H", "Gesellschaft mbH", "GmbH.", "G.m.b.H."],
        "KGaA":          ["KG a.A.", "K.G.a.A."],
        "KG":            ["K.G.", "Kommanditges."],
        "OHG":           ["O.H.G.", "Offene HG"],
        "AG":            ["A.G.", "Aktienges.", "AG."],
    }

    STREET_TEMPLATES: Final[Dict[str, List[str]]] = {
        "straße": ["str.", "strasse", "Str."],
        "gasse":  ["g.", "G."],
        "allee":  ["al.", "Al."],
        "platz":  ["pl.", "Pl."],
        "ring":   ["rng."],
        "weg":    ["w.", "Weg"],
    }

    PRODUCTS: Final[List[str]] = [
        "OsciJet Fluidik-Oszillator", "Abfüllanlage Type 3",
        "Laborabzug Pro X", "Titration Unit 5000", "Sensorik-Modul X1",
        "Industrie-Zentrifuge", "Reinraum-Schleuse V2"
    ]

    def __init__(self, seed: int = 42):
        self.fake = Faker("de_DE")
        Faker.seed(seed)
        random.seed(seed)
        
        # Sort keys by length descending to prevent partial matches (e.g., GmbH vs GmbH & Co. KG)
        self._sorted_legal_forms = sorted(self.LEGAL_FORM_VARIANTS.keys(), key=len, reverse=True)

    @staticmethod
    def _inject_typo(text: str, intensity: int = 1) -> str:
        chars = list(text)
        for _ in range(intensity):
            if len(chars) < 3: break
            idx = random.randint(1, len(chars) - 2)
            op = random.choice(["swap", "drop", "dup"])
            if op == "swap":
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            elif op == "drop":
                chars.pop(idx)
            else:
                chars.insert(idx, chars[idx])
        return "".join(chars)

    def _abbreviate_legal_form(self, text: str) -> str:
        for form in self._sorted_legal_forms:
            if form in text:
                return text.replace(form, random.choice(self.LEGAL_FORM_VARIANTS[form]), 1)
        return text

    @classmethod
    def _abbreviate_street(cls, address: str) -> str:
        addr_lower = address.lower()
        for word, variants in cls.STREET_TEMPLATES.items():
            if word in addr_lower:
                start_idx = addr_lower.find(word)
                return address[:start_idx] + random.choice(variants) + address[start_idx + len(word):]
        return address

    @staticmethod
    def _shuffle_digits(text: str) -> str:
        digits = [(i, c) for i, c in enumerate(text) if c.isdigit()]
        if len(digits) >= 2:
            idx1, idx2 = random.sample(range(len(digits)), 2)
            pos1, pos2 = digits[idx1][0], digits[idx2][0]
            chars = list(text)
            chars[pos1], chars[pos2] = chars[pos2], chars[pos1]
            return "".join(chars)
        return text

    def generate_base_data(self, n: int = 1000) -> pd.DataFrame:
        dataset = [
            {
                "id": i,
                "text": f"{self.fake.company()} | {self.fake.street_address()}, {self.fake.city()} | {random.choice(self.PRODUCTS)}"
            }
            for i in range(n)
        ]
        return pd.DataFrame(dataset)

    def generate_noisy_duplicates(self, df_base: pd.DataFrame) -> pd.DataFrame:
        noisy_records = []
        
        # Define strategy weights
        strategies = ["easy", "medium", "hard", "extreme"]
        weights = [0.20, 0.40, 0.25, 0.15]

        for _, row in df_base.iterrows():
            parts = [p.strip() for p in row["text"].split("|")]
            difficulty = random.choices(strategies, weights=weights)[0]

            if difficulty == "easy":
                target_idx = random.randint(0, len(parts) - 1)
                parts[target_idx] = self._inject_typo(parts[target_idx], intensity=random.randint(1, 2))

            elif difficulty == "medium":
                parts[0] = self._abbreviate_legal_form(parts[0])
                if len(parts) > 1:
                    parts[1] = self._shuffle_digits(self._abbreviate_street(parts[1]))
                if len(parts) > 2 and random.random() < 0.4:
                    parts[2] = parts[2].swapcase()

            elif difficulty == "hard":
                if len(parts) > 1:
                    parts.pop(random.randint(0, len(parts) - 1))
                target_idx = random.randint(0, len(parts) - 1)
                parts[target_idx] = self._inject_typo(parts[target_idx], intensity=2)

            elif difficulty == "extreme":
                parts[0] = self._abbreviate_legal_form(parts[0])
                if len(parts) > 1:
                    parts[1] = self._abbreviate_street(parts[1])
                random.shuffle(parts)

            noisy_records.append({
                "ground_truth_id": row["id"],
                "text": " | ".join(parts),
            })

        return pd.DataFrame(noisy_records)

if __name__ == "__main__":
    generator = SAPMasterDataGenerator(seed=42)
    base_df = generator.generate_base_data(n=10)
    noisy_df = generator.generate_noisy_duplicates(base_df)
    print(noisy_df.head())