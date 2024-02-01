#!/bin/bash

rm -rf __pycache__

apptainer run --nv -B /home/space/datasets/toxic_comment:/home/space/datasets/toxic_comment pml.sif python main.py explain

