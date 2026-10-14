# Reproduces findings F-01, F-02, F-03 and F-10 of PROJECT_MASTER.md (2026-10-13) without Python.
# Run from any folder:  powershell -ExecutionPolicy Bypass -File verify_findings.ps1
param(
  [string]$Root = (Resolve-Path "$PSScriptRoot\..\..").Path,
  [double]$Lat = 29.9697,
  [double]$Lon = 35.0642
)

function D2R($d){ $d*[math]::PI/180 }
function R2D($r){ $r*180/[math]::PI }

# NOAA solar-position approximation (true elevation, no refraction); within ~0.01 deg of pvlib SPA
function Get-Elevation([datetime]$utc){
  $jd = ($utc - [datetime]'2000-01-01T12:00:00').TotalDays + 2451545.0
  $jc = ($jd - 2451545.0)/36525.0
  $L = (280.46646 + $jc*(36000.76983 + $jc*0.0003032)) % 360
  $M = 357.52911 + $jc*(35999.05029 - 0.0001537*$jc)
  $e = 0.016708634 - $jc*(0.000042037 + 0.0000001267*$jc)
  $C = [math]::Sin((D2R $M))*(1.914602 - $jc*(0.004817 + 0.000014*$jc)) + [math]::Sin((D2R (2*$M)))*(0.019993 - 0.000101*$jc) + [math]::Sin((D2R (3*$M)))*0.000289
  $al = $L + $C - 0.00569 - 0.00478*[math]::Sin((D2R (125.04 - 1934.136*$jc)))
  $mo = 23 + (26 + (21.448 - $jc*(46.815 + $jc*(0.00059 - $jc*0.001813)))/60)/60
  $oc = $mo + 0.00256*[math]::Cos((D2R (125.04 - 1934.136*$jc)))
  $dec = [math]::Asin([math]::Sin((D2R $oc))*[math]::Sin((D2R $al)))
  $y = [math]::Pow([math]::Tan((D2R ($oc/2))),2)
  $eqt = 4*(R2D ($y*[math]::Sin(2*(D2R $L)) - 2*$e*[math]::Sin((D2R $M)) + 4*$e*$y*[math]::Sin((D2R $M))*[math]::Cos(2*(D2R $L)) - 0.5*$y*$y*[math]::Sin(4*(D2R $L)) - 1.25*$e*$e*[math]::Sin(2*(D2R $M))))
  $tst = (($utc.Hour*60 + $utc.Minute + $utc.Second/60.0) + $eqt + 4*$Lon) % 1440
  if($tst -lt 0){ $tst += 1440 }
  $ha = $tst/4 - 180
  $cz = [math]::Sin((D2R $Lat))*[math]::Sin($dec) + [math]::Cos((D2R $Lat))*[math]::Cos($dec)*[math]::Cos((D2R $ha))
  $cz = [math]::Max(-1.0,[math]::Min(1.0,$cz))
  90 - (R2D ([math]::Acos($cz)))
}

# --- F-02: daytime mask = elevation at hour midpoint > 5 deg -------------------------------
$day = @{}
$t = [datetime]'2024-10-01T00:00:00'
while($t -le [datetime]'2025-05-31T23:00:00'){
  if((Get-Elevation $t.AddMinutes(30)) -gt 5){ $day[$t.ToString('yyyy-MM-dd HH:mm:ss')] = 1 }
  $t = $t.AddHours(1)
}
"F-02  test-window daytime hours (expect 2580): $($day.Count)"

$preds = Import-Csv "$Root\submission_evidence\08_lstm_alignment\predictions_lstm_sota_aligned.csv"

# --- F-03: evaluate.py metrics on surviving LSTM predictions ------------------------------
foreach($tg in 'indirect','direct'){ foreach($h in 1,2,3){
  $sub = $preds | Where-Object { $_.target -eq $tg -and $_.horizon -eq "$h" -and $day.ContainsKey($_.timestamp.Substring(0,19)) }
  $yt = @($sub | ForEach-Object {[double]$_.y_true}); $yp = @($sub | ForEach-Object {[double]$_.y_pred}); $n = $yt.Count
  $mean = ($yt | Measure-Object -Average).Average; $se=0; $ae=0; $st=0
  for($i=0;$i -lt $n;$i++){ $d=$yt[$i]-$yp[$i]; $se+=$d*$d; $ae+=[math]::Abs($d); $st+=($yt[$i]-$mean)*($yt[$i]-$mean) }
  $den = if($tg -eq 'indirect'){0.6279}else{3449.0}
  "F-03  {0} D+{1}: n={2} nRMSE={3:N4} MAE={4:N4} R2={5:N4}" -f $tg,$h,$n,([math]::Sqrt($se/$n)/$den*100),($ae/$n),(1-$se/$st)
}}

# --- F-01c: daytime hours with zero AC and their label hours ------------------------------
$z = $preds | Where-Object { $_.target -eq 'direct' -and $_.horizon -eq '1' -and $day.ContainsKey($_.timestamp.Substring(0,19)) -and [double]$_.y_true -eq 0 }
"F-01c zero-AC daytime hours: $($z.Count); label hours: " + (($z | Group-Object { $_.timestamp.Substring(11,2) } | ForEach-Object { "$($_.Name)h=$($_.Count)" }) -join ', ')

# --- F-01: surviving y_true at label hour H equals raw LOCAL hour H -----------------------
function Get-RawHourly([string]$file, [string]$date){
  $sum=@{}; $cnt=@{}; $rd=[IO.StreamReader]::new($file); [void]$rd.ReadLine()
  while(($l=$rd.ReadLine()) -ne $null){ if($l.StartsWith('"'+$date)){ $p=$l.Split(','); $hh=$p[0].Trim('"').Substring(11,2)
    $ac=[double]($p[16].Trim('"'))+[double]($p[20].Trim('"')); if(-not $sum.ContainsKey($hh)){$sum[$hh]=0;$cnt[$hh]=0}; $sum[$hh]+=$ac; $cnt[$hh]++ } }
  $rd.Close(); $o=@{}; foreach($k in $sum.Keys){ $o[$k]=$sum[$k]/$cnt[$k] }; $o
}
$ly=@{}; foreach($x in ($preds | Where-Object { $_.target -eq 'direct' -and $_.horizon -eq '1' })){ $ly[$x.timestamp.Substring(0,13)] = [double]$x.y_true }
foreach($c in @(@('2412.CSV','2024-12-10'),@('2505.CSV','2025-05-14'))){
  $raw = Get-RawHourly "$Root\Raw data\$($c[0])" $c[1]
  $m = 0; foreach($h in 6..14){ $hh='{0:D2}' -f $h; if([math]::Abs($raw[$hh] - $ly["$($c[1]) $hh"]) -lt 0.5){ $m++ } }
  "F-01  $($c[1]): label hours 06-14 equal to raw LOCAL hours: $m of 9"
}
