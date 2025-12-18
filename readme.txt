## wipe everything

taskkill /F /IM python.exe 2>$null
Remove-Item attendance.db, train_status.json, model.pkl -ErrorAction SilentlyContinue
Get-ChildItem dataset -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

 python reset_db.py