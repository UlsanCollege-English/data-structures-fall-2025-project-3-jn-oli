import sys
import os

# Add project root to PYTHONPATH so `flight_planner` becomes importable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)