import json, csv
from pathlib import Path

results_dir = Path('C:/Users/salos/OneDrive/Desktop/MochiRank/data/label_results')
output_path = Path('C:/Users/salos/OneDrive/Desktop/MochiRank/data/teacher_labels.csv')

all_labels = []
missing = []
errors = []

for i in range(250):
    padded = str(i).zfill(3)
    p = results_dir / f'result_{padded}.json'
    if not p.exists():
        missing.append(padded)
        continue
    try:
        with open(p, encoding='utf-8-sig') as f:
            data = json.load(f)
        if isinstance(data, list):
            all_labels.extend(data)
        elif isinstance(data, dict) and 'labels' in data:
            all_labels.extend(data['labels'])
        elif isinstance(data, dict):
            for key in ('results', 'candidates', 'scores'):
                if key in data:
                    all_labels.extend(data[key])
                    break
    except Exception as e:
        errors.append((padded, str(e)))

print(f'Loaded {len(all_labels)} labels from {250 - len(missing)} files')
if missing:
    print(f'Missing {len(missing)} batches: {missing[:10]}')
if errors:
    print(f'Parse errors in {len(errors)} files: {errors[:5]}')

output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['candidate_id','score','rationale','stratum'], extrasaction='ignore')
    writer.writeheader()
    writer.writerows(all_labels)

print(f'Saved {output_path} with {len(all_labels)} rows')
score_dist = {}
for lbl in all_labels:
    s = str(lbl.get('score', '?'))
    score_dist[s] = score_dist.get(s, 0) + 1
print('Score distribution:', dict(sorted(score_dist.items())))
