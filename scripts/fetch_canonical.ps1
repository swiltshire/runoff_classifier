<#
.SYNOPSIS
    Download one county's canonical (seamlessly reprojected) imagery chips
    from S3, using the per-county manifest written by the pipeline.

.DESCRIPTION
    Chips are stored in S3 organized by native source CRS group, not by
    county, so there's no single S3 "folder" for a given county. Instead,
    the pipeline writes a small manifest JSON per county
    (canonical_mosaic/manifests/{county}.json) listing the exact S3 keys
    that county needs. This script downloads that manifest, then downloads
    every listed chip into a local folder - no GDAL/Python needed, just the
    AWS CLI.

.PARAMETER County
    County name, e.g. "Benton" (case-insensitive, spaces removed to match
    the pipeline's safe_name() convention).

.PARAMETER OutDir
    Destination folder for downloaded chips. Defaults to
    .\data\counties\{county}\canonical_tiles\ to mirror the pipeline's own
    on-disk layout.

.PARAMETER Bucket
    S3 bucket name. Defaults to "sagemaker-gst-stage.sharing".

.PARAMETER Prefix
    S3 prefix under the bucket where canonical_mosaic/ lives. Defaults to
    "serge-wiltshire/runoff-classifier-data/canonical_mosaic".

.EXAMPLE
    .\fetch_canonical.ps1 -County Benton

.EXAMPLE
    .\fetch_canonical.ps1 -County "Cass" -OutDir C:\data\Cass\canonical_tiles
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$County,

    [string]$OutDir,

    [string]$Bucket = "sagemaker-gst-stage.sharing",

    [string]$Prefix = "serge-wiltshire/runoff-classifier-data/canonical_mosaic"
)

$ErrorActionPreference = "Stop"

# safe_name() in src/utils/indiana_cogs.py only replaces characters outside
# [A-Za-z0-9._ -] with "_" - it does NOT lowercase or strip spaces/periods.
# Every real Indiana county name (including "St. Joseph") already satisfies
# that character set, so safe_name(county) == county unchanged for all of
# them - no transformation needed here.
$CountySafe = $County

if (-not $OutDir) {
    $OutDir = Join-Path -Path (Join-Path -Path (Join-Path -Path "." "data") "counties") -ChildPath "$County\canonical_tiles"
}

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Error "AWS CLI ('aws') not found on PATH. Install it first: https://aws.amazon.com/cli/"
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$manifestKey = "$Prefix/manifests/$CountySafe.json"
$manifestUri = "s3://$Bucket/$manifestKey"
$manifestLocal = Join-Path -Path $env:TEMP -ChildPath "manifest_$CountySafe.json"

Write-Host "Fetching manifest for '$County' ($manifestUri)..."
& aws s3 cp $manifestUri $manifestLocal --only-show-errors
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to download manifest. Has this county's canonical tiles been generated yet?"
    exit 1
}

$manifest = Get-Content -Raw -Path $manifestLocal | ConvertFrom-Json
$keys = $manifest.keys
$total = $keys.Count

Write-Host "Manifest: $total chip(s), chip_size_px=$($manifest.chip_size_px), canonical_crs=$($manifest.canonical_crs)"
Write-Host "Downloading to: $OutDir"
Write-Host ""

$downloaded = 0
$skipped = 0
$failed = 0
$i = 0

foreach ($key in $keys) {
    $i++
    $filename = Split-Path -Path $key -Leaf
    $localPath = Join-Path -Path $OutDir -ChildPath $filename

    # skip-if-exists-with-matching-size, so a re-run resumes instead of
    # re-downloading everything
    if (Test-Path $localPath) {
        $remoteSize = (& aws s3api head-object --bucket $Bucket --key $key --query ContentLength --output text 2>$null)
        $localSize = (Get-Item $localPath).Length
        if ($remoteSize -and ([int64]$remoteSize -eq $localSize)) {
            $skipped++
            Write-Progress -Activity "Fetching canonical chips for $County" -Status "$i / $total (skip $filename)" -PercentComplete (($i / $total) * 100)
            continue
        }
    }

    & aws s3 cp "s3://$Bucket/$key" $localPath --only-show-errors
    if ($LASTEXITCODE -eq 0) {
        $downloaded++
    } else {
        $failed++
        Write-Warning "Failed to download $key"
    }

    Write-Progress -Activity "Fetching canonical chips for $County" -Status "$i / $total ($filename)" -PercentComplete (($i / $total) * 100)
}

Write-Progress -Activity "Fetching canonical chips for $County" -Completed

Write-Host ""
Write-Host "Done: $downloaded downloaded, $skipped already present (skipped), $failed failed, out of $total total."
if ($failed -gt 0) {
    Write-Warning "$failed chip(s) failed to download - re-run this script to retry just those."
}
