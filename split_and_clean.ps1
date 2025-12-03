# split_and_clean.ps1  ← 100% SAFE VERSION (copies only, deletes nothing)

mkdir data\young -Force
mkdir data\old   -Force

$labels = Import-Csv ffhq_aging_labels.csv

$youngCount = 0
$oldCount   = 0
$maxEach    = 1000

# Auto-find the thumbnails folder
$thumbFolder = Get-ChildItem -Directory | Where-Object { $_.Name -like "*thumbnail*" } | Select-Object -First 1
if (-not $thumbFolder) { 
    Write-Error "Thumbnails folder not found! Expected thumbnails128x128"
    exit 
}

Write-Host "Found: $($thumbFolder.Name)"
Write-Host "Copying 1000 young + 1000 old faces (nothing will be deleted)..."

foreach ($row in $labels) {
    $id = [int]$row.image_number
    $filename = "{0:D5}.png" -f $id
    $src = Join-Path $thumbFolder.FullName $filename

    if (Test-Path $src) {
        $age = [int]($row.age_group -replace '[^0-9].*','')  # e.g. "60-69" → 60

        if ($age -lt 30 -and $youngCount -lt $maxEach) {
            Copy-Item $src data\young\ -Force
            $youngCount++
        }
        elseif ($age -ge 50 -and $oldCount -lt $maxEach) {
            Copy-Item $src data\old\ -Force
            $oldCount++
        }
    }

    if ($youngCount -ge $maxEach -and $oldCount -ge $maxEach) { break }
}

Write-Host "SUCCESS! (Everything preserved)"
Write-Host "data\young → $youngCount images (copied)"
Write-Host "data\old   → $oldCount images (copied)"
Write-Host "thumbnails128x128 folder and CSV are 100% untouched and safe"