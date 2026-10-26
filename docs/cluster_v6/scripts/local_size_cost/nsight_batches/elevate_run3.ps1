$bash = 'C:\Program Files\Git\usr\bin\bash.exe'
$script = 'C:\Users\cchen\AppData\Local\Temp\claude\C--Users-cchen-PycharmProjects-vulkan-demo\fd68689d-e4b4-4b3f-8636-fbe3d1d866de\scratchpad\e8\nsight\elevated_run3.sh'
$p = Start-Process -FilePath $bash -ArgumentList @($script) -Verb RunAs -Wait -PassThru
exit $p.ExitCode
