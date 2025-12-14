#!/bin/sh

out=out.log

nohup sh eicl.sh 0 > $out 2>&1 &

tail -f $out
