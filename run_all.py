import subprocess
import sys
scripts = ["generate_tracks.py", "player_mapping.py"]
for script in scripts:
    print(f"\n=== Running {script} ===")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error running {script}:")
        print(result.stderr)
        sys.exit(result.returncode)
print("\nAll steps completed successfully.") 