$ErrorActionPreference = "Continue"
$BASE = "http://DESKTOP-7AGI2SN:8000"
$results = @()

function Log($name, $status, $detail) {
    $script:results += [PSCustomObject]@{Endpoint=$name; Status=$status; Detail=$detail}
    if ($status -eq "PASS") { Write-Host "  PASS  $name" -ForegroundColor Green }
    elseif ($status -eq "WARN") { Write-Host "  WARN  $name - $detail" -ForegroundColor Yellow }
    else { Write-Host "  FAIL  $name - $detail" -ForegroundColor Red }
}

function WriteJson($data) {
    [System.IO.File]::WriteAllText("$env:TEMP\_test_req.json", $data)
}

function Post($path, $body) {
    WriteJson $body
    return curl.exe -s -X POST "$BASE$path" -H "Content-Type: application/json" -d "@$env:TEMP\_test_req.json" 2>$null
}
function AuthPost($path, $body, $token) {
    WriteJson $body
    return curl.exe -s -X POST "$BASE$path" -H "Content-Type: application/json" -H "Authorization: Bearer $token" -d "@$env:TEMP\_test_req.json" 2>$null
}
function Get_($path) {
    return curl.exe -s "$BASE$path" 2>$null
}
function AuthGet($path, $token) {
    return curl.exe -s "$BASE$path" -H "Authorization: Bearer $token" 2>$null
}
function AuthPut($path, $body, $token) {
    WriteJson $body
    return curl.exe -s -X PUT "$BASE$path" -H "Content-Type: application/json" -H "Authorization: Bearer $token" -d "@$env:TEMP\_test_req.json" 2>$null
}
function AuthDel($path, $token) {
    return curl.exe -s -X DELETE "$BASE$path" -H "Authorization: Bearer $token" 2>$null
}

function IsOk($resp) {
    return ($resp -and -not ($resp -match '"detail"'))
}

Write-Host "`n====== BACKEND ENDPOINT SMOKE TEST ======`n" -ForegroundColor Cyan

# ── 1. Health & Metrics ──
Write-Host "`n--- Health & Metrics ---" -ForegroundColor Cyan
$h = Get_ "/health"
if ($h -match "ok|healthy") { Log "/health" "PASS" "" } else { Log "/health" "FAIL" $h }
$m = Get_ "/metrics"
if ($m -match "python_info") { Log "/metrics" "PASS" "" } else { Log "/metrics" "FAIL" "no prometheus data" }

# ── 2. Auth ──
Write-Host "`n--- Auth ---" -ForegroundColor Cyan

$cLogin = Post "/auth/login" '{"username":"citizen1","password":"citizen123"}' | ConvertFrom-Json
if ($cLogin.access_token) { Log "POST /auth/login (citizen)" "PASS" ""; $CT = $cLogin.access_token; $CRT = $cLogin.refresh_token } else { Log "POST /auth/login (citizen)" "FAIL" "$($cLogin.detail)" }

$aLogin = Post "/auth/login" '{"username":"admin","password":"admin123"}' | ConvertFrom-Json
if ($aLogin.access_token) { Log "POST /auth/login (admin)" "PASS" ""; $AT = $aLogin.access_token } else { Log "POST /auth/login (admin)" "FAIL" "$($aLogin.detail)" }

# Try officer login - may have different password in seeded DB
$oLogin = Post "/auth/login" '{"username":"officer1","password":"officer123"}' | ConvertFrom-Json
if ($oLogin.access_token) { $OT = $oLogin.access_token; Log "POST /auth/login (officer)" "PASS" "" }
else {
    # Try fetching officers via admin and use admin as officer fallback
    Log "POST /auth/login (officer)" "WARN" "officer1 creds rejected, using admin token for officer tests"
    $OT = $AT
}

$me = AuthGet "/auth/me" $CT | ConvertFrom-Json
if ($me.username -eq "citizen1") { Log "GET /auth/me" "PASS" "" } else { Log "GET /auth/me" "FAIL" "$me" }

$meUpdate = AuthPut "/auth/me" '{"full_name":"Rajesh Kumar Swain"}' $CT
if (IsOk $meUpdate) { Log "PUT /auth/me" "PASS" "" } else { Log "PUT /auth/me" "FAIL" $meUpdate }

$officers = AuthGet "/auth/officers" $CT | ConvertFrom-Json
if ($officers -is [array]) { Log "GET /auth/officers" "PASS" "found $($officers.Count) officers" } else { Log "GET /auth/officers" "WARN" "$officers" }

$refresh = Post "/auth/refresh" "{`"refresh_token`":`"$CRT`"}" | ConvertFrom-Json
if ($refresh.access_token) { Log "POST /auth/refresh" "PASS" ""; $CT = $refresh.access_token } else { Log "POST /auth/refresh" "FAIL" "$($refresh.detail)" }

# ── 3. Grievances (citizen) ──
Write-Host "`n--- Grievances (citizen) ---" -ForegroundColor Cyan

$newG = AuthPost "/grievances" '{"title":"Test Water Supply Issue","description":"No water supply in Ward 5 for 3 days. Residents are suffering.","category":"water_supply","location":"Bhubaneswar, Ward 5","district":"khordha","block":"bhubaneswar"}' $CT | ConvertFrom-Json
if ($newG.id) {
    Log "POST /grievances" "PASS" "id=$($newG.id), tracking=$($newG.tracking_number)"
    $GID = $newG.id
    $GTRACK = $newG.tracking_number
} else { Log "POST /grievances" "FAIL" "$($newG.detail)"; $GID = $null }

$gList = AuthGet "/grievances" $CT | ConvertFrom-Json
if ($gList -is [array]) { Log "GET /grievances" "PASS" "$($gList.Count) grievances" } else { Log "GET /grievances" "FAIL" "$gList" }

if ($GID) {
    $gDetail = AuthGet "/grievances/$GID" $CT | ConvertFrom-Json
    if ($gDetail.id -eq $GID) { Log "GET /grievances/{id}" "PASS" "" } else { Log "GET /grievances/{id}" "FAIL" "$gDetail" }
}

if ($GTRACK) {
    $gTrack = Get_ "/grievances/track/$GTRACK"
    if ($gTrack -match "tracking_number") { Log "GET /grievances/track/{num}" "PASS" "" } else { Log "GET /grievances/track/{num}" "FAIL" $gTrack }
}

if ($GID) {
    $fb = AuthPost "/grievances/$GID/feedback" '{"rating":4,"comment":"Good response time"}' $CT
    if (IsOk $fb) { Log "POST /grievances/{id}/feedback" "PASS" "" } else { Log "POST /grievances/{id}/feedback" "WARN" $fb }
}

# ── 4. Grievances (officer) ──
Write-Host "`n--- Grievances (officer/admin) ---" -ForegroundColor Cyan

$oList = AuthGet "/grievances" $OT | ConvertFrom-Json
if ($oList -is [array]) { Log "GET /grievances (officer)" "PASS" "$($oList.Count) grievances" } else { Log "GET /grievances (officer)" "FAIL" "$oList" }

if ($GID) {
    $note = AuthPost "/grievances/$GID/notes" '{"note":"Inspected the area, issue confirmed."}' $OT
    if (IsOk $note) { Log "POST /grievances/{id}/notes" "PASS" "" } else { Log "POST /grievances/{id}/notes" "WARN" $note }

    $statusUp = AuthPut "/grievances/$GID/status" '{"status":"in_progress"}' $OT
    if (IsOk $statusUp) { Log "PUT /grievances/{id}/status" "PASS" "" } else { Log "PUT /grievances/{id}/status" "WARN" $statusUp }

    $assign = AuthPut "/grievances/$GID/assign" '{"officer_id":"test"}' $OT
    if (IsOk $assign) { Log "PUT /grievances/{id}/assign" "PASS" "" } else { Log "PUT /grievances/{id}/assign" "WARN" $assign }

    $resolve = AuthPut "/grievances/$GID/resolve" '{"resolution":"Water supply restored after fixing the main pipeline.","officer":"Test Officer","add_to_service_memory":false}' $OT
    if (IsOk $resolve) { Log "PUT /grievances/{id}/resolve" "PASS" "" } else { Log "PUT /grievances/{id}/resolve" "WARN" $resolve }
}

$multiDept = AuthGet "/grievances/multi-department" $OT
Log "GET /grievances/multi-department" $(if ($multiDept -and -not ($multiDept -match '"detail"')) { "PASS" } else { "WARN" }) $multiDept

$schemeMatched = AuthGet "/grievances/scheme-matched" $OT
Log "GET /grievances/scheme-matched" $(if ($schemeMatched -and -not ($schemeMatched -match '"detail"')) { "PASS" } else { "WARN" }) ""

# ── 5. Admin ──
Write-Host "`n--- Admin ---" -ForegroundColor Cyan

$users = AuthGet "/admin/users" $AT | ConvertFrom-Json
if ($users -is [array]) { Log "GET /admin/users" "PASS" "$($users.Count) users" } else { Log "GET /admin/users" "FAIL" "$users" }

$deadlines = AuthGet "/admin/deadline-alerts" $AT
Log "GET /admin/deadline-alerts" $(if (IsOk $deadlines) { "PASS" } else { "WARN" }) ""

$spam = AuthGet "/admin/spam-flagged" $AT
Log "GET /admin/spam-flagged" $(if ($spam -and -not ($spam -match '"detail":"Not')) { "PASS" } else { "WARN" }) ""

$anomalies = AuthGet "/admin/officer-anomalies" $AT
Log "GET /admin/officer-anomalies" $(if ($anomalies -and -not ($anomalies -match '"detail":"Not')) { "PASS" } else { "WARN" }) ""

$reports = curl.exe -s "$BASE/admin/reports" -H "Authorization: Bearer $AT" 2>$null
Log "GET /admin/reports" $(if ($reports -match "<html|html>") { "PASS" } else { "WARN" }) ""

$reportStats = AuthGet "/admin/reports/stats" $AT
Log "GET /admin/reports/stats" $(if (IsOk $reportStats) { "PASS" } else { "WARN" }) ""

# ── 6. Knowledge ──
Write-Host "`n--- Knowledge ---" -ForegroundColor Cyan

$schemes = AuthGet "/knowledge/schemes" $OT
Log "GET /knowledge/schemes" $(if ($schemes -and -not ($schemes -match '"detail":"Not')) { "PASS" } else { "WARN" }) ""

$docs = AuthGet "/knowledge/documentation" $OT
Log "GET /knowledge/documentation" $(if ($docs -and -not ($docs -match '"detail":"Not')) { "PASS" } else { "WARN" }) ""

$smem = AuthGet "/knowledge/service-memory" $OT
Log "GET /knowledge/service-memory" $(if ($smem -and -not ($smem -match '"detail":"Not')) { "PASS" } else { "WARN" }) ""

$schemeSearch = AuthPost "/knowledge/schemes/search" '{"query":"water supply scheme"}' $OT
Log "POST /knowledge/schemes/search" $(if ($schemeSearch -and -not ($schemeSearch -match '"detail":"Not')) { "PASS" } else { "WARN" }) ""

# ── 7. Analytics ──
Write-Host "`n--- Analytics ---" -ForegroundColor Cyan

$analytics = AuthGet "/analytics" $OT
Log "GET /analytics" $(if (IsOk $analytics) { "PASS" } else { "WARN" }) ""

$geo = AuthGet "/analytics/geographical" $OT
Log "GET /analytics/geographical" $(if (IsOk $geo) { "PASS" } else { "WARN" }) ""

# ── 8. Public ──
Write-Host "`n--- Public ---" -ForegroundColor Cyan

$pubG = Get_ "/public/grievances"
Log "GET /public/grievances" $(if ($pubG -and -not ($pubG -match '"detail":"Not')) { "PASS" } else { "WARN" }) ""

# ── 9. Chat ──
Write-Host "`n--- Chat ---" -ForegroundColor Cyan

$chat = AuthPost "/chat" '{"message":"What is the status of water supply in Bhubaneswar?","session_id":"test-session-1"}' $CT
Log "POST /chat" $(if ($chat -and -not ($chat -match '"detail":"Not')) { "PASS" } else { "WARN" }) ""

# ── 10. Other ──
Write-Host "`n--- Other ---" -ForegroundColor Cyan

$loc = Get_ "/location/from-ip"
Log "GET /location/from-ip" $(if ($loc) { "PASS" } else { "WARN" }) ""

$forecasts = AuthGet "/forecasts/latest" $OT
Log "GET /forecasts/latest" $(if ($forecasts -and -not ($forecasts -match '"detail":"Not')) { "PASS" } else { "WARN" }) ""

$forecastAll = AuthGet "/forecasts" $OT
Log "GET /forecasts" $(if ($forecastAll -and -not ($forecastAll -match '"detail":"Not')) { "PASS" } else { "WARN" }) ""

$systemic = AuthGet "/systemic-issues" $OT
Log "GET /systemic-issues" $(if ($systemic -and -not ($systemic -match '"detail":"Not')) { "PASS" } else { "WARN" }) ""

$spamStatus = AuthGet "/auth/spam-status" $CT
Log "GET /auth/spam-status" $(if ($spamStatus -and -not ($spamStatus -match '"detail":"Not')) { "PASS" } else { "WARN" }) ""

# ── Summary ──
Write-Host "`n====== SUMMARY ======" -ForegroundColor Cyan
$pass = ($results | Where-Object { $_.Status -eq "PASS" }).Count
$warn = ($results | Where-Object { $_.Status -eq "WARN" }).Count
$fail = ($results | Where-Object { $_.Status -eq "FAIL" }).Count
Write-Host "  PASS: $pass  |  WARN: $warn  |  FAIL: $fail  |  TOTAL: $($results.Count)" -ForegroundColor White

if ($fail -gt 0) {
    Write-Host "`nFailed endpoints:" -ForegroundColor Red
    $results | Where-Object { $_.Status -eq "FAIL" } | ForEach-Object { Write-Host "  $($_.Endpoint): $($_.Detail)" -ForegroundColor Red }
}
if ($warn -gt 0) {
    Write-Host "`nWarnings:" -ForegroundColor Yellow
    $results | Where-Object { $_.Status -eq "WARN" } | ForEach-Object { Write-Host "  $($_.Endpoint): $($_.Detail)" -ForegroundColor Yellow }
}

Write-Host "`nDone.`n"
