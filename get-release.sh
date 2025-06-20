#!/bin/bash

while read listfile; do archivename="${listfile%.txt}.tar.gz"; size=$(du -cb $(cat "$listfile") | grep total$ | awk '{print $1}'); tar -cv -T "$listfile" -f - | pv -s "$size" | pigz > "$archivename"; done < archive.txt