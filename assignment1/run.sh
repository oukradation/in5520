#! /bin/bash
# img d theta window iso

python3 img_process.py $1 $2 $3 $4 IDM $5 &
python3 img_process.py $1 $2 $3 $4 inertia $5 &
python3 img_process.py $1 $2 $3 $4 shade $5 &
