#!/bin/bash

# script.sh is a script to run Docker for testing.  This is called by CI like
# "bash .flexci/linux/script.sh torch15".
#
# Environment variables:
# - PPE_FLEXCI_IMAGE_NAME ... The Docker image name (without tag) to be
#       used for CI.
# - DRYRUN ... Set DRYRUN=1 for local testing.  This disables destructive
#       actions and make the script print commands.

# Fail immedeately on error or unbound variables.
set -eu

# note: Docker image names can be overridden per project using secret environment
# variables of FlexCI.
PPE_FLEXCI_IMAGE_NAME=${PPE_FLEXCI_IMAGE_NAME:-asia.gcr.io/pfn-public-ci/pytorch-pfn-extras-ci}

################################################################################
# Main function
################################################################################
main() {
  TARGET="$1"
  SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE}")/../.."; pwd)"

  # Initialization.
  prepare_docker &
  wait

  # Prepare docker args.
  docker_args=(
    docker run --rm --ipc=host --privileged --runtime=nvidia
    --volume="${SRC_ROOT}:/src"
    --workdir="/src"
  )

  # Run target-specific commands.
  case "${TARGET}" in
    torch* )
      # Unit test.
      run "${docker_args[@]}" \
          "${PPE_FLEXCI_IMAGE_NAME}:${TARGET}" \
          /src/.flexci/linux/unittest.sh "${TARGET}"
      ;;
    prep )
      # Build and push docker images for unit tests.
      run "${SRC_ROOT}/.flexci/linux/build_and_push.sh" \
          "${PPE_FLEXCI_IMAGE_NAME}"
      ;;
    * )
      echo "${TARGET}: Invalid target."
      exit 1
      ;;
  esac
}

################################################################################
# Utility functions
################################################################################

# run executes a command.  If DRYRUN is enabled, run just prints the command.
run() {
  echo '+' "$@" >&2
  if [ "${DRYRUN:-}" == '' ]; then
    "$@"
  fi
}

# Configure docker to pull images from gcr.io.
prepare_docker() {
  run gcloud auth configure-docker
}

################################################################################
# Bootstrap
################################################################################
main "$@"
