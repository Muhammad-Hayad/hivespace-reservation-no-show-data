
import numpy as np
import pandas as pd

# Make results reproducible
np.random.seed(42)

# Number of reservations
NUM_RESERVATIONS = 1000

# Generate member IDs
member_ids = [
    f"MEM{np.random.randint(1, 251):04d}"
    for _ in range(NUM_RESERVATIONS)
]

# Generate reservation dates
start_date = "2026-01-01"
end_date = "2026-11-30"

reservation_dates = pd.to_datetime(
    np.random.choice(
        pd.date_range(start_date, end_date),
        size=NUM_RESERVATIONS
    )
)

# Generate time slots
time_slots = np.random.choice(
    ["Morning", "Afternoon", "Evening"],
    size=NUM_RESERVATIONS,
    p=[0.35, 0.40, 0.25]
)

# Generate member tenure (days)
member_tenure_days = np.random.randint(
    1, 1201, size=NUM_RESERVATIONS
)

# Assign no-show probabilities
# Newer members have a somewhat higher probability
# 35% for brand-new members, decreasing to 25% over one year
tenure_factor = np.minimum(member_tenure_days, 365) / 365

base_probability = 0.35 - (tenure_factor * 0.10)

# Add variation based on time slot
time_adjustment = np.select(
    [
        time_slots == "Morning",
        time_slots == "Afternoon",
        time_slots == "Evening"
    ],
    [0.02, -0.02, 0.03],
    default=0
)

# Add randomness so the relationship is not too perfect
random_noise = np.random.normal(
    loc=0,
    scale=0.07,
    size=NUM_RESERVATIONS
)

no_show_probability = np.clip(
    base_probability + time_adjustment + random_noise,
    0.03,
    0.65
)

# Generate binary no-show outcomes
no_show = np.random.binomial(
    n=1,
    p=no_show_probability
)

# Create the dataset
df = pd.DataFrame({
    "member_id": member_ids,
    "reservation_date": reservation_dates,
    "time_slot": time_slots,
    "member_tenure_days": member_tenure_days,
    "no_show": no_show
})

# Save dataset
df.to_csv("reservations.csv", index=False)

# Display dataset information
print("HiveSpace reservation dataset generated successfully!")
print(f"\nTotal reservations: {len(df)}")
print("\nDataset preview:")
print(df.head())

print("\nNo-show distribution:")
print(df["no_show"].value_counts())
print("\nNo-show percentages:")
print(df["no_show"].value_counts(normalize=True) * 100)

print("\nAverage no-show rate by tenure group:")
df["tenure_group"] = pd.cut(
    df["member_tenure_days"],
    bins=[0, 90, 365, 1200],
    labels=["New", "Intermediate", "Long-term"]
)
print(df.groupby("tenure_group", observed=False)["no_show"].mean())
