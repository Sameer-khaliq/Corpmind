from corpmind.agents.report import generate_report
import json

# tumhare load_test.py ka final result use karo (jo real_state pe ainvoke() se aaya)
report = generate_report(result)   # 'result' woh hi hai jo run_batch() se mila tha
print(json.dumps(report, indent=2))