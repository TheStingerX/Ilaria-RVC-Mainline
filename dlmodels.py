import subprocess

urls = [
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D32k.pth",
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k.pth",
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D48k.pth",
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G32k.pth",
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G40k.pth",
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G48k.pth",
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D32k.pth",
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth",
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth",
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G32k.pth",
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth",
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth"
]


commands = [
    "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -d /content/Ilaria-RVC-Mainline/assets/hubert -o hubert_base.pt",
    "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://raw.githubusercontent.com/poiqazwsx/Ilaria-RVC-Mainline/main/assets/hubert/hubert_inputs.pth -d /content/Ilaria-RVC-Mainline/assets/hubert -o hubert_inputs.pth",
    "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt -d /content/Ilaria-RVC-Mainline/assets/rmvpe -o rmvpe.pt",
    "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/poiqazwsx/Ilaria-RVC-Mainline/blob/main/assets/rmvpe/rmvpe_inputs.pth -d /content/Ilaria-RVC-Mainline/assets/rmvpe -o rmvpe.pth"
]

for url in urls:
    filename = url.split("/")[-1]
    command = f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M {url} -d /content/Ilaria-RVC-Mainline/assets/pretrained_v2 -o {filename}"
    subprocess.run(command, shell=True)


for command in commands:
    subprocess.run(command, shell=True)
