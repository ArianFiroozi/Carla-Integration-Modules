import subprocess

def run_powershell_script(script_path):
    try:
        result = subprocess.run(['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', script_path], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    script_path = "loopback_adapter.ps1"
    run_powershell_script(script_path)
