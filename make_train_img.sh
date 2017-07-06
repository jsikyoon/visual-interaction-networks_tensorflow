#!/bin/bash

for i in {0..9}
do
  python physics_engine.py --prefix=$i
done
