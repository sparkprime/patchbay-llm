#!/bin/bash

DIRS="./tests ./patchbay_llm ./examples"

isort $DIRS
black $DIRS
