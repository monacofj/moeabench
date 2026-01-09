import subprocess

format = "png"
target_folder = "."

cmd = [
    "pyreverse",
    "-o", format,
    "-p", "EVO_NEW",
    "-a","10",
    "C:\\MoeaBench\\MoeaBench"
]

print("Executando:", " ".join(cmd))
subprocess.run(cmd, check=True)