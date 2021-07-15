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

# note: These values can be overridden per project using secret environment
# variables of FlexCI.
PPE_FLEXCI_IMAGE_NAME=${PPE_FLEXCI_IMAGE_NAME:-asia.gcr.io/pfn-public-ci/pytorch-pfn-extras-ci}
PPE_FLEXCI_GCS_BUCKET=${PPE_FLEXCI_GCS_BUCKET:-chainer-artifacts-pfn-public-ci}

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
    --env CUDA_VISIBLE_DEVICES
    --volume="${SRC_ROOT}:/src"
    --volume="/tmp/output:/output"
    --workdir="/src"
  )

  # Run target-specific commands.
  case "${TARGET}" in
    torch* )
      # Unit test.
      .flexci/linux/download_mnist.sh
      run "${docker_args[@]}" \
          "${PPE_FLEXCI_IMAGE_NAME}:${TARGET}" \
          /src/.flexci/linux/unittest.sh "${TARGET}"
      gsutil -m -q cp /tmp/output/pysen.txt gs://${PPE_FLEXCI_GCS_BUCKET}/pytorch-pfn-extras/pysen/${CI_JOB_ID}/pysen.txt
      echo "pysen output: https://storage.cloud.google.com/${PPE_FLEXCI_GCS_BUCKET}/pytorch-pfn-extras/pysen/${CI_JOB_ID}/pysen.txt"
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
