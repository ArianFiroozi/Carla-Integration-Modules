$timeout = 60  # Seconds without heartbeat before restart
$pythonExe = "C:/Users/H/anaconda3/envs/torch-gpu/python.exe"
$scriptPath = "c:/Users/H/Desktop/IOT/Carla-Integration-Modules/env.py"
$heartbeatFile = "heartbeat.txt"

function Start-Training {
    # Start process with output in current window
    $process = Start-Process -FilePath $pythonExe -ArgumentList $scriptPath -PassThru -NoNewWindow
    Write-Host "Started training with PID $($process.Id)"
    return $process
}

while ($true) {
    # Cleanup previous heartbeat
    if (Test-Path $heartbeatFile) { Remove-Item $heartbeatFile }

    # Start training process
    $trainingProcess = Start-Training
    
    # Monitoring loop
    $active = $true
    while ($active) {
        Start-Sleep -Seconds 10
        
        # Check if process exists
        if ($trainingProcess.HasExited) {
            Write-Host "Process exited with code $($trainingProcess.ExitCode)"
            $active = $false
            break
        }

        # Check heartbeat
        if (Test-Path $heartbeatFile) {
            $lastWrite = (Get-Item $heartbeatFile).LastWriteTime
            $age = (Get-Date) - $lastWrite
            if ($age.TotalSeconds -gt $timeout) {
                Write-Host "No heartbeat for $($age.TotalSeconds) seconds - restarting..."
                Stop-Process -Id $trainingProcess.Id -Force
                $active = $false
            }
        } else {
            Write-Host "Waiting for initial heartbeat..."
        }
    }

    # Cleanup
    if (-not $trainingProcess.HasExited) {
        Stop-Process -Id $trainingProcess.Id -Force -ErrorAction SilentlyContinue
    }

    # Cool-down before restart
    Write-Host "Restarting in 5 seconds..."
    Start-Sleep -Seconds 5
}