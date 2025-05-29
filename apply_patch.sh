#!/bin/bash

SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
mv patches/core.py $SITE_PACKAGES/slices/