#!/bin/sh

# Function to check and set environment variables
set_env_vars() {
  if [ "$(uname)" = "Darwin" ]; then
    # macOS specific env:
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  elif [ "$(uname)" != "Linux" ]; then
    echo "Unsupported operating system."
    exit 1
  fi
}

# Function to activate or create virtual environment
handle_venv() {
  printf "Checking for virtual environment..."
  if [ -d ".venv" ]; then
    printf " Found.\nActivating venv..."
    . .venv/bin/activate
    printf " Done.\n"
  else
    printf " Not found.\n"
    requirements_file="requirements.txt"

    pyenv_exists=0
    if command -v pyenv >/dev/null 2>&1; then
      pyenv_exists=1
    fi

    # Check if pyenv is installed and version 3.8 is available
    pyenv_v38_installed=0
    if [ $pyenv_exists -eq 1 ]; then
      if pyenv versions --bare | grep -q "3.8"; then
        pyenv_v38_installed=1
      fi
    fi

    python_v38_exists=0
    if command -v python3.8 >/dev/null 2>&1; then
      python_v38_exists=1
    fi


    # Check if Python 3.8 is installed
    printf "Checking for Python 3.8..."
    if ! [ $pyenv_v38_installed -eq 1 ] && ! [ $python_v38_exists -eq 1 ]; then
      printf " Not found.\nInstalling Python 3.8..."
      if [ "$(uname)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
        echo "Using Homebrew..."
        brew install python@3.8
      elif [ "$(uname)" = "Linux" ]; then
        if command -v apt-get >/dev/null 2>&1; then
          echo "Using apt..."
          sudo apt-get update
          sudo apt-get install python3.8
        elif command -v pacman >/dev/null 2>&1; then
          echo "Using pacman..."
          sudo pacman -Syu python38
        elif command -v dnf >/dev/null 2>&1; then
          echo "Using dnf..."
          sudo dnf install python38
        else
          echo "Unsupported package manager for automatic Python 3.8 installation."
          echo "Please install Python 3.8 manually."
          exit 1
        fi
      else
        echo "Unsupported operating system for automatic Python 3.8 installation."
        echo "Please install Python 3.8 manually."
        exit 1
      fi
    fi
    printf " Found.\n"

    printf "Creating venv..."
    python3.8 -m venv .venv
    . .venv/bin/activate
    printf " Done.\n"

    # update pip
    printf "Updating pip..."
    python3.8 -m pip install --upgrade pip > /dev/null 2>> pkgerr.log
    printf " Done.\n"

    # Check if required packages are installed and install them if not
    echo "Checking for required packages..."
    if [ -f "${requirements_file}" ]; then
      installed_packages=$(python3.8 -m pip freeze)
      while IFS= read -r package; do
        expr "${package}" : "^#.*" > /dev/null && continue
        package_name=$(echo "${package}" | sed 's/[<>=!].*//')
        if ! echo "${installed_packages}" | grep -q "${package_name}"; then
          printf "%s not found. Installing..." "${package_name}"
          python3.8 -m pip install --upgrade "${package}" > /dev/null 2>> pkgerr.log
          printf " Done.\n"
        fi
      done < "${requirements_file}"
    else
      echo "${requirements_file} not found. Please ensure the requirements file with required packages exists."
      exit 1
    fi

    if [ -s pkgerr.log ]; then
      echo "Something happened whilst installing packages. Please check pkgerr.log for more details in case of failure."
    fi
  fi
}

# Function to download models
download_models() {
  echo "Checking if models are downloaded..."
  chmod +x tools/dlmodels.sh
  ./tools/dlmodels.sh
  echo "Models downloaded."

  if [ $? -ne 0 ]; then
    exit 1
  fi
}

# Function to run the main script
run_main_script() {
  printf "%0.s=" $(seq 1 "$(tput cols)")
  message="Running main script with args: $passargs"
  printf "%*s\n" $(((${#message}+$(tput cols))/2)) "$message"
  printf "%0.s=" $(seq 1 "$(tput cols)")
  python3.8 infer-web.py --pycmd python3.8 $passargs
}

# Parse command-line arguments
passargs=""

while getopts ":p:" opt; do
  case ${opt} in
    p)
      passargs=$OPTARG
      ;;
    \?)
      echo "Invalid option: $OPTARG" 1>&2
      ;;
    :)
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))

# Call functions
set_env_vars
handle_venv
download_models
run_main_script
