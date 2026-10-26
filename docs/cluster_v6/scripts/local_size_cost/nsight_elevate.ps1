param([string]$label, [string]$case, [string]$start, [string]$dur)
$bash = 'C:\Program Files\Git\usr\bin\bash.exe'
$script = 'C:\Users\cchen\AppData\Local\Temp\claude\C--Users-cchen-PycharmProjects-vulkan-demo\fd68689d-e4b4-4b3f-8636-fbe3d1d866de\scratchpad\e8\nsight\capture.sh'
$p = Start-Process -FilePath $bash -ArgumentList @($script, $label, $case, $start, $dur) -Verb RunAs -Wait -PassThru
exit $p.ExitCode
