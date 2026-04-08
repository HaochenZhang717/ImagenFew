#!/usr/bin/env bash
set -euo pipefail

echo "=== Partition Summary ==="
sinfo -o "%P %.10a %.5D %.10T %.12G %.20N"
echo

echo "=== Running/Pending GPU Jobs ==="
squeue -o "%.18i %.12P %.20j %.10u %.2t %.12M %.8D %R" || true
echo

echo "=== Node GPU State ==="
scontrol show node | awk '
BEGIN {
  node=""; gres=""; alloc=""; state="";
}
/^NodeName=/ {
  if (node != "") {
    print node "\tstate=" state "\tgres=" gres "\talloc=" alloc;
  }
  node=$1; sub("NodeName=","",node);
  gres=""; alloc=""; state="";
}
/State=/ {
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^State=/) {
      state=$i; sub("State=","",state);
    }
    if ($i ~ /^Gres=/) {
      gres=$i; sub("Gres=","",gres);
    }
    if ($i ~ /^AllocTRES=/) {
      alloc=$i; sub("AllocTRES=","",alloc);
    }
  }
}
END {
  if (node != "") {
    print node "\tstate=" state "\tgres=" gres "\talloc=" alloc;
  }
}'
echo

echo "=== My Jobs Only ==="
squeue -u "${USER}" -o "%.18i %.12P %.20j %.2t %.12M %.8D %R" || true
