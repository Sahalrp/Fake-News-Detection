import pandas as pd
import random
from datetime import datetime, timedelta

# Topics for diversity
TOPICS = [
    "Politics", "Science", "Sports", "Health", "Technology", "Finance", "Entertainment", "World", "Environment"
]

# Template sentences for fake and real articles
FAKE_TEMPLATES = [
    "BREAKING: {event} causes chaos in {place}. Experts warn of unprecedented consequences.",
    "Shocking discovery: {person} involved in {scandal}. Public reacts strongly.",
    "Fake: {event} will change the world forever, insiders claim.",
    "Rumors spread about {company} launching {product} next month.",
    "Unverified report: {country} to ban {thing} starting {date}."
]

REAL_TEMPLATES = [
    "{event} successfully completed by {person} in {place}.",
    "Official report: {company} announces {product} with new features.",
    "{country} celebrates progress in {topic} with new initiatives.",
    "Research confirms {finding} in {field}.",
    "{event} held in {place} draws international attention."
]

EVENTS = ["Elections", "Space Mission", "Olympics", "Climate Summit", "Tech Conference", "Medical Breakthrough", "Stock Market Rally", "Film Festival"]
PLACES = ["New York", "London", "Beijing", "Sydney", "Paris", "Berlin", "Tokyo", "Dubai"]
PEOPLE = ["Dr. Smith", "President Lee", "Elon Musk", "Serena Williams", "Angela Merkel", "Prof. Chen", "Oprah Winfrey", "Cristiano Ronaldo"]
COMPANIES = ["TechCorp", "MediLife", "EcoWorld", "FinBank", "Sportify", "EduPlus"]
PRODUCTS = ["AI Assistant", "Vaccine", "Electric Car", "Streaming Service", "Smartphone", "Fitness App"]
COUNTRIES = ["USA", "India", "Germany", "Brazil", "Australia", "Japan", "Canada", "France"]
THINGS = ["plastic bags", "cryptocurrency", "gasoline cars", "single-use bottles"]
FINDINGS = ["water on Mars", "new species of bird", "record low unemployment", "vaccine efficacy", "AI outperforming humans"]
FIELDS = ["astronomy", "biology", "economics", "medicine", "technology"]
SCANDALS = [
    "financial fraud", "doping scandal", "data breach", "tax evasion", "illegal lobbying",
    "insider trading", "academic plagiarism", "match-fixing", "bribery case", "embezzlement"
]

START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2024, 4, 1)


def random_date():
    delta = END_DATE - START_DATE
    random_days = random.randint(0, delta.days)
    return (START_DATE + timedelta(days=random_days)).strftime("%Y-%m-%d")

def generate_article(template, topic):
    return template.format(
        event=random.choice(EVENTS),
        place=random.choice(PLACES),
        person=random.choice(PEOPLE),
        company=random.choice(COMPANIES),
        product=random.choice(PRODUCTS),
        country=random.choice(COUNTRIES),
        thing=random.choice(THINGS),
        date=random_date(),
        finding=random.choice(FINDINGS),
        field=random.choice(FIELDS),
        topic=topic,
        scandal=random.choice(SCANDALS)
    )

def generate_synthetic_news(n_per_topic=1200):
    fake_rows = []
    real_rows = []
    for topic in TOPICS:
        for _ in range(n_per_topic):
            # Fake
            fake_template = random.choice(FAKE_TEMPLATES)
            fake_title = fake_template.split(":")[0][:60]
            fake_text = generate_article(fake_template, topic)
            fake_rows.append({
                "title": fake_title,
                "text": fake_text,
                "subject": topic,
                "date": random_date()
            })
            # Real
            real_template = random.choice(REAL_TEMPLATES)
            real_title = real_template.split(":")[0][:60]
            real_text = generate_article(real_template, topic)
            real_rows.append({
                "title": real_title,
                "text": real_text,
                "subject": topic,
                "date": random_date()
            })
    fake_df = pd.DataFrame(fake_rows)
    real_df = pd.DataFrame(real_rows)
    fake_df.to_csv("synthetic_fake.csv", index=False)
    real_df.to_csv("synthetic_true.csv", index=False)
    print(f"Generated {len(fake_df)} fake and {len(real_df)} real synthetic articles.")

if __name__ == "__main__":
    generate_synthetic_news(n_per_topic=1200)  # You can increase this number if needed
