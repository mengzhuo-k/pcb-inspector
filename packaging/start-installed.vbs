Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
command = """" & root & "\runtime\pythonw.exe"" """ & root & "\launcher.py"""
If WScript.Arguments.Count > 0 Then
    If WScript.Arguments(0) = "--stop" Then command = command & " --stop"
End If
shell.Run command, 0, False
