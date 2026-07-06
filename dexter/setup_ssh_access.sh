#!/usr/bin/env bash
# Generate Ed25519 SSH key with passphrase for Dexter server access.
#
# Usage:
#   bash dexter/setup_ssh_access.sh
#   bash dexter/setup_ssh_access.sh --email you@example.com
#
# Send the .pub file contents to the server admin (reply to their email).

set -euo pipefail

EMAIL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --email)
      EMAIL="${2:-}"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash dexter/setup_ssh_access.sh [--email you@example.com]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

KEY_DIR="${HOME}/.ssh"
KEY_PATH="${KEY_DIR}/dexter_ed25519"
COMMENT="dexter-access${EMAIL:+ ${EMAIL}}"

mkdir -p "${KEY_DIR}"
chmod 700 "${KEY_DIR}"

if [[ -f "${KEY_PATH}" ]]; then
  echo "Key already exists: ${KEY_PATH}"
  echo "Public key:"
  cat "${KEY_PATH}.pub"
  echo ""
  echo "To create a new key, rename or remove ${KEY_PATH} first."
  exit 0
fi

echo "Generating Ed25519 key (you will be asked for a passphrase — required by admin)."
echo "Key path: ${KEY_PATH}"
echo ""

ssh-keygen -t ed25519 -b 256 -f "${KEY_PATH}" -C "${COMMENT}"

chmod 600 "${KEY_PATH}"
chmod 644 "${KEY_PATH}.pub"

echo ""
echo "=== Done ==="
echo "Send this public key to the server admin:"
echo "------------------------------------------------------------------------"
cat "${KEY_PATH}.pub"
echo "------------------------------------------------------------------------"
echo ""
echo "Connect after admin adds the key:"
echo "  ssh -i ${KEY_PATH} <user>@<dexter-host>"
echo ""
echo "Optional ~/.ssh/config entry:"
echo "  Host dexter"
echo "    HostName <dexter-host>"
echo "    User <user>"
echo "    IdentityFile ${KEY_PATH}"
