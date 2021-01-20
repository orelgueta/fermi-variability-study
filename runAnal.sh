#!/bin/bash

wdir=$1;
FERMICONFIG=$2
CONFIGANAL=$3

source activate fermi
cd $wdir;
python $wdir/anal.py 'analyze' $FERMICONFIG $CONFIGANAL

exit
