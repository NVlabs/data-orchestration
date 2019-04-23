#!/bin/tcsh -f

#source ~/mytools_workspace/whoop/functional-model/util/setup_whoop_env.csh
limit coredumpsize 0

set PROJECT = research_arch_misc
set CPUS    = 1

if( $1 == "small" ) then
  set QUEUE = o_cpu_8G_1H
  set TILE_SIZE = 16
  set GDIR = small
endif 

if( $1 == "medium" ) then
  set QUEUE = o_cpu_16G_4H
  set TILE_SIZE = 256
  set GDIR = medium
endif 

if( $1 == "big" ) then
  set QUEUE = o_cpu_48G_24H
  set TILE_SIZE = 262144
  set GDIR = big
endif 

mkdir outputs >& /dev/null

setenv NOCOMPILE 1

set get_stats = 0
if( $#argv == 3 ) then
  set get_stats = 1
endif

foreach g ( `ls ../data-sets/input/$GDIR/*.in-offsets | grep $2 | grep -v sym | sed "s/.*\///g" | sed "s/.in-off.*//g"` )

  set gname = `basename $g`
  printf "$gname "

  # if( $get_stats == 0 ) then
  #   printf "\n"
  # endif

  printf "\n"

  ## setenv stats_file_name ../outputs/$GDIR/$gname.stats_untiled_nobuffers.txt
  ## cd tiled-compressed 
  ## if( $get_stats ) then
  ##   $PYTHON ../util/stats_parser.py $stats_file_name
  ## else
  ##   echo " --> untiled-nobuffers"
  ##   qsub -P $PROJECT -n $CPUS -q $QUEUE -oo $stats_file_name:r.lsf.out ./run.dst-src.sh $g -1 -1 $stats_file_name
  ## endif
  ## cd ../
  
  setenv stats_file_name ../outputs/$GDIR/$gname.stats_untiled_withbuffers.txt
  cd untiled-compressed 
  if( $get_stats ) then
    $PYTHON ../util/stats_parser.py $stats_file_name
  else
    echo " --> untiled-withbuffers"
    qsub -P $PROJECT -n $CPUS -q $QUEUE -oo $stats_file_name:r.lsf.out ./run.sh $g $stats_file_name
  endif
  cd ../
 
  setenv stats_file_name ../outputs/$GDIR/$gname.stats_1d_tiled.txt
  cd tiled-compressed 
  if( $get_stats ) then
    $PYTHON ../util/stats_parser.py $stats_file_name
  else
    echo " --> 1d tiled"
    qsub -P $PROJECT -n $CPUS -q $QUEUE -oo $stats_file_name:r.lsf.out ./run.src-dst.sh $g $TILE_SIZE -1 $stats_file_name
  endif
  cd ../
 
  setenv stats_file_name ../outputs/$GDIR/$gname.stats_2d_src-dst_tiled.txt
  cd tiled-compressed 
  if( $get_stats ) then
    $PYTHON ../util/stats_parser.py $stats_file_name
  else
    echo " --> src and dst tiled"
    qsub -P $PROJECT -n $CPUS -q $QUEUE -oo $stats_file_name:r.lsf.out ./run.src-dst.sh $g $TILE_SIZE $TILE_SIZE $stats_file_name
  endif
  cd ../
  
  setenv stats_file_name ../outputs/$GDIR/$gname.stats_2d_dst-src_tiled.txt
  cd tiled-compressed 
  if( $get_stats ) then
    $PYTHON ../util/stats_parser.py $stats_file_name
  else
    echo " --> src and dst tiled"
    qsub -P $PROJECT -n $CPUS -q $QUEUE -oo $stats_file_name:r.lsf.out ./run.dst-src.sh $g $TILE_SIZE $TILE_SIZE $stats_file_name
  endif
  cd ../
 
    echo ""
end

