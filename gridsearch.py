import json
import subprocess
import os

log_filename = "gridsearch.log"

for i in range(21):
	threshold = 0 + i * 10

	env = dict(os.environ, PICNIC_BASKET1_PRICE_MARGIN_THRESHOLD2=str(threshold))

	# Append grid search info to the log
	with open(log_filename, "a") as log_file:
		log_file.write(f"Grid Search On threshold {threshold}\n")

	# Run the external command and capture output
	cmd = ["prosperity3bt", "Trader.py", "2"]
	proc = subprocess.run(cmd, env= env, capture_output=True, text=True)

	output_lines = proc.stdout.strip().splitlines()
	for line in output_lines[-7:-1]:
		with open(log_filename, "a") as log_file:
			log_file.write(line + "\n")