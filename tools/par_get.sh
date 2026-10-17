#!/bin/bash
# Parallel ranged download (20 MB parts, 4 at a time, retries) of URL to OUT; size in bytes given. Parts already in OUT.parts are kept;
# the parts are joined only when all are complete. Usage: par_get.sh URL SIZE OUT
URL=$1; SIZE=$2; OUT=$3; P=$OUT.parts; mkdir -p $P; CH=20971520; N=$(( (SIZE + CH - 1) / CH ))
get() { k=$1; a=$((k*CH)); b=$((a+CH-1)); [ $b -ge $SIZE ] && b=$((SIZE-1)); f=$P/$(printf %04d $k); [ -s $f ] && return 0
  for t in $(seq 1 40); do curl -s -L --max-time 900 -r $a-$b -o $f.tmp "$URL" && [ $(stat -c %s $f.tmp) -eq $((b-a+1)) ] && mv $f.tmp $f && return 0; sleep 15; done; echo "part $k failed"; }
export -f get; export URL SIZE CH P
seq 0 $((N-1)) | xargs -P 4 -I{} bash -c 'get {}'
if [ $(ls $P | grep -vc tmp) -eq $N ]; then rm -f $P/*.tmp; cat $(ls $P/[0-9][0-9][0-9][0-9] | sort) > $OUT && rm -rf $P && md5sum $OUT; else echo "incomplete: $(ls $P | grep -vc tmp) of $N parts"; fi
