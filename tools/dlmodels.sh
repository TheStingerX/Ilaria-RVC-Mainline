#!/bin/sh

printf "Checking for required commands..."
if command -v curl > /dev/null 2>&1
then
    printf " Done.\n"
else
    printf " Failed.\n"
    echo "curl command not found. Please install curl and try again."
    exit 1
fi

echo "Checking for required directories and files..."
check_dir() {
  printf "Checking %s..." "$1"
  if [ -d "$1" ]; then
      printf " Found.\n"
  else
      printf " Not found.\nCreating..."
      mkdir -p "$1"
      [ -d "$1" ] && printf " Done.\n" || printf " Failed.\n"
  fi
}

check_dir "./assets/pretrained"
check_dir "./assets/pretrained_v2"
check_dir "./assets/uvr5_weights"
check_dir "./assets/uvr5_weights/onnx_dereverb_By_FoxJoy"

echo "Checking for required files..."
check_file_pretrained() {
  printf "Checking %s..." "$2"
  if [ -f "./assets/""$1""/""$2""" ]; then
      printf " Found.\n"
  else
      printf " Not found.\nChecking for curl..."
      if command -v curl > /dev/null 2>&1; then
          printf " Found.\nDownloading %s from huggingface..." "$2"
          curl -L https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"$1"/"$2" -o ./assets/"$1"/"$2" > /dev/null 2>> moderr.log
          [ -f "./assets/""$1""/""$2""" ] && printf " Done.\n" || printf " Failed.\n"
      else
          printf " Not found.\n"
          echo "Please install curl and try again."
          exit 1
      fi
  fi
}

check_file_special() {
  printf "Checking %s..." "$2"
  if [ -f "./assets/""$1""/""$2""" ]; then
      printf " Found.\n"
  else
      printf " Not found.\nChecking for curl..."
      if command -v curl > /dev/null 2>&1; then
          printf " Found.\nDownloading %s from huggingface..." "$2"
          curl -L https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"$2" -o ./assets/"$1"/"$2" > /dev/null 2>> moderr.log
          [ -f "./assets/""$1""/""$2""" ] && printf " Done.\n" || printf " Failed.\n"
      else
          printf " Not found.\n"
          echo "Please install curl and try again."
          exit 1
      fi
  fi
}

check_file_pretrained pretrained D32k.pth
check_file_pretrained pretrained D40k.pth
check_file_pretrained pretrained D48k.pth
check_file_pretrained pretrained G32k.pth
check_file_pretrained pretrained G40k.pth
check_file_pretrained pretrained G48k.pth
check_file_pretrained pretrained_v2 f0D40k.pth
check_file_pretrained pretrained_v2 f0G40k.pth
check_file_pretrained pretrained_v2 D40k.pth
check_file_pretrained pretrained_v2 G40k.pth
check_file_pretrained uvr5_weights HP2_all_vocals.pth
check_file_pretrained uvr5_weights HP3_all_vocals.pth
check_file_pretrained uvr5_weights HP5_only_main_vocal.pth
check_file_pretrained uvr5_weights VR-DeEchoAggressive.pth
check_file_pretrained uvr5_weights VR-DeEchoDeReverb.pth
check_file_pretrained uvr5_weights VR-DeEchoNormal.pth
check_file_pretrained uvr5_weights "onnx_dereverb_By_FoxJoy/vocals.onnx"
check_file_special rmvpe rmvpe.pt
check_file_special hubert hubert_base.pt
