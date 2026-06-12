import csv
from pathlib import Path

path = Path('C:/Users/salos/OneDrive/Desktop/MochiRank/data/teacher_labels.csv')
VALID = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}

rows = list(csv.DictReader(path.open(encoding='utf-8')))
fixed = 0
for r in rows:
    v = float(r['score'])
    nearest = min(VALID, key=lambda x: abs(x - v))
    if nearest != v:
        r['score'] = str(nearest)
        fixed += 1
    else:
        r['score'] = str(v)

with open(path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['candidate_id', 'score', 'rationale', 'stratum'])
    w.writeheader()
    w.writerows(rows)
print(f'Normalized {fixed} scores. Total rows: {len(rows)}')
