import requests

url = "http://127.0.0.1:8000/predict"

# Single test
ticket = {
    "subject": "Payment issue",
    "description": "Charged twice this month on my card."
}
response = requests.post(url, json=ticket)
print("Single Test Response:", response.json())

# Batch test (one per category)
samples = [
    {"subject": "App crashes", "description": "The app closes on startup."},
    {"subject": "Add dark mode", "description": "We need a dark mode option."},
    {"subject": "Server down", "description": "503 error across all services."},
    {"subject": "Billing problem", "description": "Charged $50 instead of $40."},
    {"subject": "Password reset", "description": "Reset email not arriving."}
]

print("\n--- Batch Tests ---")
for t in samples:
    res = requests.post(url, json=t)
    print(f"Input: {t['subject']} → Predicted: {res.json()}")
