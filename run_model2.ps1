$timeout = 30  # Seconds without heartbeat before restart
$pythonExe = "C:/Users/H/anaconda3/envs/torch-gpu/python.exe"
$scriptPath = "c:/Users/H/Desktop/IOT/Carla-Integration-Modules/env.py"
$heartbeatFile = "heartbeat.txt"
$exePath = "C:\Users\H\Desktop\Carla\CarlaUE4.exe"
$port = 2000

function Start-Training {
    # Start process with output in current window
    $process = Start-Process -FilePath $pythonExe -ArgumentList $scriptPath -PassThru -NoNewWindow
    Write-Host "Started training with PID $($process.Id)"
    return $process
}

function Check-Port {
    param (
        [int]$port
    )
    $portInUse = (Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue) -ne $null
    return $portInUse
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

    # Check if port 2000 is not in use and run .exe file
    if (-not (Check-Port -port $port)) {
        Write-Host "Port $port is not in use. Running $exePath..."
        Start-Process -FilePath $exePath -NoNewWindow 
        Start-Sleep -Seconds 30 
    } else {
        Write-Host "Port $port is in use. Skipping $exePath execution."
    }

    # Cool-down before restart
    Write-Host "Restarting in 5 seconds..."
    Start-Sleep -Seconds 5
}
