#! /bin/bash
# img d theta window iso

python img_process2.py $1 $2 $3 $4 IDM $5 &
python img_process2.py $1 $2 $3 $4 inertia $5 &
python img_process2.py $1 $2 $3 $4 shade $5 &
