"""Flood the prediction endpoint with requests.

Interactive UI:   locust -f locust/locustfile.py --host https://<your-app>.onrender.com
Headless run:     locust -f locust/locustfile.py --host http://localhost:8080 --headless \
                         -u 50 -r 10 -t 60s --csv locust/results/c1_u50
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

TEST_DIR = Path(__file__).resolve().parent.parent / "data" / "test"
IMAGES = [p.read_bytes() for p in sorted(TEST_DIR.rglob("*.JPG")) + sorted(TEST_DIR.rglob("*.jpg"))][:60]


class LeafUser(HttpUser):
    wait_time = between(0.5, 1.5)          # each simulated user sends ~1 request per second

    @task(10)
    def predict(self):
        img = random.choice(IMAGES)
        self.client.post("/predict", files={"file": ("leaf.jpg", img, "image/jpeg")}, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
