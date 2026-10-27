$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$taskOutput = Join-Path $taskRoot 'outputs/voice-samples'
New-Item -ItemType Directory -Force $taskOutput | Out-Null
$taskVoice = New-Object -ComObject SAPI.SpVoice
$taskToken = New-Object -ComObject SAPI.SpObjectToken
$taskToken.SetId('HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens\MSTTS_V110_jaJP_HarukaM')
$taskVoice.Voice = $taskToken
$taskStream = New-Object -ComObject SAPI.SpFileStream
try {
    $taskStream.Open((Join-Path $taskOutput 'base-haruka.wav'), 3, $false)
    $taskVoice.AudioOutputStream = $taskStream
    $taskVoice.Rate = -2
    $taskVoice.Volume = 90
    [void]$taskVoice.Speak('あ、あの……こんにちは。望月京子です。えっと、今日は……少しだけ、お話ししても、いいですか？', 0)
} finally {
    $taskStream.Close()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskStream)
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskVoice)
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskToken)
}
Write-Output '日本語のベース音声を生成しました。'
