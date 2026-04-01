import random
import pandas as pd
from faker import Faker


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
            record = f"{self.fake.company()} | {self.fake.street_address()}, {self.fake.city()} | {random.choice(self.products)}"
            data.append({"id": i, "text": record})
        return pd.DataFrame(data)
    
    def generate_noisy_duplicates(self, df_base: pd.DataFrame) -> pd.DataFrame:
        noisy_data = []
        for _, row in df_base.iterrows():
            text = row["text"]

            if random.random() < 0.4: text = text.replace('GmbH', 'Ges.m.b.H')
            if random.random() < 0.4: text = text.replace('AG', 'A.G.')
            if random.random() < 0.3: text = text.replace('OsciJet', 'Oscijet')
            if random.random() < 0.2: text = text.replace(' | ', ' ') 
            
            if random.random() < 0.5 and len(text) > 10:
                idx = random.randint(0, len(text)-3)
                text = text[:idx] + text[idx+1] + text[idx] + text[idx+2:]
                
            noisy_data.append({"ground_truth_id": row['id'], "text": text})
        return pd.DataFrame(noisy_data)
    
DataGenerator = SAPMasterDataGenerator()