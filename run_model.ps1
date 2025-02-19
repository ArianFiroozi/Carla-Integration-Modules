$successCount = 0
$totalRuns = 300
$scriptPath = "C:/Users/H/Desktop/IOT/Carla-Integration-Modules/env.py"
$pythonExecutable = "C:/Users/H/anaconda3/envs/torch-gpu/python.exe"

while ($successCount -lt $totalRuns) {
    $startTime = Get-Date

    # Start the Python process with a timeout of 60 seconds
    $process = Start-Process -FilePath $pythonExecutable -ArgumentList $scriptPath -PassThru
    $process.WaitForExit(80000)

    $endTime = Get-Date
    $runtime = ($endTime - $startTime).TotalSeconds

    if ($process.HasExited -and $runtime -lt 80 -and $runtime -gt 20) {
        $successCount++
        Write-Output "Run $successCount was successful (completed in $runtime seconds)"
    } else {
        # Kill the process if it is still running
        if (!$process.HasExited) {
            $process.Kill()
        }
        Write-Output "Run failed or timed out"
    }
}

Write-Output "Completed $successCount successful runs."
