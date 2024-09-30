# https://stackoverflow.com/questions/9948517/how-to-stop-a-powershell-script-on-the-first-error

trap {
  Write-Host "###Trap start###"
  Write-Host $_
  Write-Host "###Trap end###"
}

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSDefaultParameterValues['*:ErrorAction']='Stop'

function RunOrDie {
    $cmd, $params = $args
    $params = @($params)
    $global:LastExitCode = 0
    & $cmd @params
    if (-not $?) {
        throw "Command failed (exit code = $LastExitCode): $cmd $params"
    }
}

function RunOrDieWithRetry {
    $retry, $cmd, $params = $args
    for ($i = 1; $i -le $retry; $i++) {
        try {
            RunOrDie $cmd $params
            return
        } catch {
            $errmsg = $error[0]
            Write-Host "RunOrDieWithRetry (attempt ${i}): ${errmsg}"
        }
    }
    throw "No more retry."
}
