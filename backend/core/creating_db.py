import pandas as pd
import bcrypt
import random
from faker import Faker
import uuid

# Initialize faker with Indian locale
fake = Faker("en_IN")

# Sample Indian coastal / flood-prone cities
cities = [
    "Mumbai", "Chennai", "Kolkata", "Visakhapatnam", "Kochi",
    "Mangalore", "Puri", "Paradip", "Tuticorin", "Gopalpur",
    "Cuddalore", "Thiruvananthapuram", "Nagapattinam", "Guwahati",
    "Bhubaneswar"
]

# Roles
roles = ["official", "citizen"]

# Function to generate hashed password
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

# Generate synthetic dataset
def generate_users(n=200):
    users = []
    for _ in range(n):
        name = fake.name()
        uid = str(uuid.uuid4())  # unique user_id

        # Specify probability distribution: e.g., 20% officials, 80% citizens
        weights = [0.2, 0.8]

        role = random.choices(roles, weights = weights, k = 1)[0]

        # Generate a simple password (name + random digits)
        raw_password = name.split()[0].lower() + str(random.randint(100, 999))
        hashed_pwd = hash_password(raw_password)

        user = {
            "user_name": name,
            "user_id": uid,
            "role": role,
            "password": raw_password,   # plain password (for testing only!)
            "hashed_pwd": hashed_pwd,
        }
        users.append(user)
    return users

if __name__ == "__main__":
    data = generate_users(200)  # generate 50 users
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv("user_data.csv", index=False)
    print("user_data.csv generated with", len(df), "rows")
