#!/bin/bash

set -oexu pipefail

echo '[INFO] Compiling report...'

pandoc ./docs/assignment/report.md -o ./docs/assignment/report.pdf

echo '[INFO] Compiling slides...'

pandoc -t beamer ./docs/assignment/slides.md -o ./docs/assignment/slides.pdf
