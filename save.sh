#!/usr/bin/env bash

./build.sh

docker save laibi_algorithm:latest | gzip -c > laibi_algorithm.tar.gz