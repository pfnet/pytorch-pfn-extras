#!/bin/bash

# script.sh is a script to run Docker for testing.  This is called by CI like
# "bash .flexci/linux/script.sh torch15".
#
# Environment variables:
# - PPE_FLEXCI_GCS_BUCKET ... The bucket to save htmlcov.
# - PPE_FLEXCI_IMAGE_NAME ... The Docker image name (without tag) to be
#       used for CI.
# - PPE_FLEXCI_IMAGE_REBUILD ... Whether to force rebuilding the docker image
#       from scratch ("1") or try reusing the docker image previously pushed
#       to the registry ("0", default).
# - PPE_FLEXCI_IMAGE_PUSH ... Whether to push docker image to the registry
#       for future reuse ("1", default) or not ("0").
# - DRYRUN ... Set DRYRUN=1 for local testing.  This disables destructive
#       actions and make the script print commands.

# Fail immedeately on error or unbound variables.
set -eu

# note: These values can be overridden per project using secret environment
# variables of FlexCI.
export PPE_FLEXCI_GCS_BUCKET=${PPE_FLEXCI_GCS_BUCKET:-chainer-artifacts-pfn-public-ci}
export PPE_FLEXCI_IMAGE_NAME=${PPE_FLEXCI_IMAGE_NAME:-asia-northeast1-docker.pkg.dev/pfn-artifactregistry/tmp-public-ci-dlfw/pytorch-pfn-extras-ci}
export PPE_FLEXCI_IMAGE_REBUILD="${PPE_FLEXCI_IMAGE_REBUILD:-0}"
export PPE_FLEXCI_IMAGE_PUSH="${PPE_FLEXCI_IMAGE_PUSH:-1}"

################################################################################
# Main function
################################################################################
main() {
  TARGET="$1"
  TEST_MODE="${2:-}"
  SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE}")/../.."; pwd)"

  echo "[PPE CI] TARGET: ${TARGET}"
  echo "[PPE CI] TEST_MODE: ${TEST_MODE}"
  echo "[PPE CI] SRC_ROOT: ${SRC_ROOT}"
  echo "[PPE CI] PPE_FLEXCI_GCS_BUCKET: ${PPE_FLEXCI_GCS_BUCKET}"
  echo "[PPE CI] PPE_FLEXCI_IMAGE_NAME: ${PPE_FLEXCI_IMAGE_NAME}"
  echo "[PPE CI] PPE_FLEXCI_IMAGE_REBUILD: ${PPE_FLEXCI_IMAGE_REBUILD}"
  echo "[PPE CI] PPE_FLEXCI_IMAGE_PUSH: ${PPE_FLEXCI_IMAGE_PUSH}"

  # Initialization.
  gcloud auth configure-docker asia-northeast1-docker.pkg.dev

  # Prepare docker images.
  run "${SRC_ROOT}/.flexci/linux/build_and_push.sh" "${TARGET}"

  # Prepare docker args.
  docker_args=(
    docker run --rm --ipc=host --privileged --runtime=nvidia
    --env CUDA_VISIBLE_DEVICES
    --volume="${SRC_ROOT}:/src"
    --volume="/tmp/output:/output"
    --workdir="/src"
  )

  # Run unit test.
  run .flexci/linux/download_mnist.sh
  run "${docker_args[@]}" \
      "${PPE_FLEXCI_IMAGE_NAME}:${TARGET}" \
      /src/.flexci/linux/unittest.sh "${TEST_MODE}"
  if [ "${TEST_MODE}" == "unittest" ]; then
    run gsutil -m -q cp -r /tmp/output/htmlcov gs://${PPE_FLEXCI_GCS_BUCKET}/pytorch-pfn-extras/pytest-cov/${CI_JOB_ID}/htmlcov
    echo "pytest-cov output: https://storage.googleapis.com/${PPE_FLEXCI_GCS_BUCKET}/pytorch-pfn-extras/pytest-cov/${CI_JOB_ID}/htmlcov/index.html"
  fi
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

################################################################################
# Bootstrap
################################################################################
main "$@"
