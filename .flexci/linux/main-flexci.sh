#!/bin/bash

# Bootstrap script for FlexCI.

set -ue

echo "Environment Variables:"
env

pull_req=""
if [[ "${FLEXCI_BRANCH:-}" == refs/pull/* ]]; then
    # Extract pull-request ID
    pull_req="$(echo "${FLEXCI_BRANCH}" | cut -d/ -f3)"
    echo "Testing Pull-Request: #${pull_req}"
fi

export PPE_FLEXCI_IMAGE_PUSH="0"
if [[ "${pull_req}" == "" ]]; then
    # Push images when running on branch.
    export PPE_FLEXCI_IMAGE_PUSH="1"
fi

"$(dirname ${0})/script.sh" "${@}"
