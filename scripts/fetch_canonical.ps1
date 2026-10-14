<#
.SYNOPSIS
    Download one county's canonical (seamlessly reprojected) imagery chips
    from S3, using the per-county manifest written by the pipeline.
    Data is organized by state, so the state is required.

.DESCRIPTION
    Chips are stored in S3 organized by native source CRS group, not by
    county, so there's no single S3 "folder" for a given county. Instead,
    the pipeline writes a small manifest JSON per county
    ({State}/canonical_mosaic/manifests/{county}.json) listing the exact S3 keys
    that county needs. This script downloads that manifest, then downloads
    every listed chip into a local folder - no GDAL/Python needed, just the
    AWS CLI.

.PARAMETER State
    Two-letter state code, e.g. "IN". Required: many county names (Lake,
    Clark, Jackson, ...) exist in more than one state.

.PARAMETER County
    County name exactly as the pipeline names it, e.g. "Benton" or
    "St. Joseph".

.PARAMETER OutDir
    Destination folder for downloaded chips. Defaults to
    .\data\{State}\counties\{county}\canonical_tiles\ to mirror the
    pipeline's own on-disk layout.

.PARAMETER Bucket
    S3 bucket name. Defaults to "sagemaker-gst-stage.sharing".

.PARAMETER Root
    S3 prefix under the bucket where the per-state folders live. Defaults to
    "serge-wiltshire/runoff-classifier-data".

.EXAMPLE
    .\fetch_canonical.ps1 -State IN -County Benton

.EXAMPLE
    .\fetch_canonical.ps1 -State IN -County "Cass" -OutDir C:\data\Cass\canonical_tiles
#>

param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z]{2}$')]
    [string]$State,

    [Parameter(Mandatory = $true)]
    [string]$County,

    [string]$OutDir,

    [string]$Bucket = "sagemaker-gst-stage.sharing",

    [string]$Root = "serge-wiltshire/runoff-classifier-data"
)

$ErrorActionPreference = "Stop"

$State = $State.ToUpper()
$Prefix = "$Root/$State/canonical_mosaic"

# safe_name() in src/utils/common.py only replaces characters outside
# [A-Za-z0-9._ -] with "_" - it does NOT lowercase or strip spaces/periods.
# Every configured county name (e.g. "St. Joseph", "De Witt") already
# satisfies that character set, so no transformation is needed here.
$CountySafe = $County

if (-not $OutDir) {
    $OutDir = Join-Path -Path "." -ChildPath "data\$State\counties\$County\canonical_tiles"
}

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Error "AWS CLI ('aws') not found on PATH. Install it first: https://aws.amazon.com/cli/"
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$manifestKey = "$Prefix/manifests/$CountySafe.json"
$manifestUri = "s3://$Bucket/$manifestKey"
$manifestLocal = Join-Path -Path $env:TEMP -ChildPath "manifest_${State}_$CountySafe.json"

Write-Host "Fetching manifest for $State / '$County' ($manifestUri)..."
& aws s3 cp $manifestUri $manifestLocal --only-show-errors
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to download manifest. Has this county's canonical tiles been generated yet?"
    exit 1
}

$manifest = Get-Content -Raw -Path $manifestLocal | ConvertFrom-Json
$keys = $manifest.keys
$total = $keys.Count

# Refuse to download anything from a manifest that belongs to a different
# state/county or points outside this state's canonical prefix.
if ($manifest.state -ne $State) {
    Write-Error "Manifest state '$($manifest.state)' does not match -State $State."
    exit 1
}
if ($manifest.county -ne $County) {
    Write-Error "Manifest county '$($manifest.county)' does not match -County '$County'."
    exit 1
}
$outside = @($keys | Where-Object { -not $_.StartsWith("$Prefix/") })
if ($outside.Count -gt 0) {
    Write-Error "Manifest lists $($outside.Count) key(s) outside $Prefix/ (e.g. $($outside[0]))."
    exit 1
}

Write-Host "Manifest: $State/$County, $total chip(s), chip_size_px=$($manifest.chip_size_px), canonical_crs=$($manifest.canonical_crs)"
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
