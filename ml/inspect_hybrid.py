import joblib
from pathlib import Path
import json

p = Path('..') / 'artifacts' / 'hybrid.pkl'
if not p.exists():
    p = Path('artifacts') / 'hybrid.pkl'
if not p.exists():
    print('hybrid.pkl not found')
    raise SystemExit(2)

obj = joblib.load(p)
print('Type:', type(obj))
try:
    # If it's a dict-like
    keys = list(obj.keys()) if hasattr(obj, 'keys') else None
    print('Keys:', keys)
    metrics = obj.get('metrics') if isinstance(obj, dict) else None
    print('Metrics:', json.dumps(metrics, indent=2))
except Exception as e:
    print('Error inspecting object:', e)
