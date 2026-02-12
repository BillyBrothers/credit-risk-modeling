import requests
import json

# Server URL
BASE_URL = "http://localhost:8000"

# Test 1: Health check
print("=" * 60)
print("TEST 1: Health Check")
print("=" * 60)
response = requests.get(f"{BASE_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 2: Score ONE applicant
print("\n" + "=" * 60)
print("TEST 2: Score Single Applicant (Expected: APPROVE)")
print("=" * 60)

applicant_1 = {
    "person_age": 25,
    "person_income": 60000,
    "person_home_ownership": "RENT",
    "person_emp_length": 3.0,
    "loan_intent": "EDUCATION",
    "loan_grade": "B",
    "loan_amnt": 5000,
    "loan_int_rate": 9.5,
    "loan_percent_income": 0.08,
    "cb_person_default_on_file": False,
    "cb_person_cred_hist_length": 5
}

response = requests.post(f"{BASE_URL}/score", json=applicant_1)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 3: Score another applicant (Expected: DENY or MANUAL_REVIEW)
print("\n" + "=" * 60)
print("TEST 3: Score Different Applicant (Higher Risk)")
print("=" * 60)

applicant_2 = {
    "person_age": 45,
    "person_income": 30000,
    "person_home_ownership": "RENT",
    "person_emp_length": 0.5,
    "loan_intent": "PERSONAL",
    "loan_grade": "F",
    "loan_amnt": 25000,
    "loan_int_rate": 20.0,
    "loan_percent_income": 0.83,
    "cb_person_default_on_file": True,
    "cb_person_cred_hist_length": 2
}

response = requests.post(f"{BASE_URL}/score", json=applicant_2)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 4: Batch score multiple applicants
print("\n" + "=" * 60)
print("TEST 4: Batch Score Three Applicants")
print("=" * 60)

batch_applicants = [applicant_1, applicant_2, applicant_1]  # Mix of applicants

response = requests.post(f"{BASE_URL}/batch-score", json=batch_applicants)
print(f"Status: {response.status_code}")
print(f"Number of results: {len(response.json())}")
for i, result in enumerate(response.json()):
    print(f"\n  Applicant {i}: {result['approval']['decision']}")

print("\n" + "=" * 60)
print("All tests complete!")
print("=" * 60)