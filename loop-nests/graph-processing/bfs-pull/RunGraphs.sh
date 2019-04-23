#!/bin/tcsh -f

#source ~/mytools_workspace/whoop/functional-model/util/setup_whoop_env.csh
limit coredumpsize 0

set PROJECT = research_arch_misc
set CPUS    = 1
set QUEUE   = o_cpu_8G_24H   #o_cpu_16G_24H

set TILE_SIZE = 32

mkdir outputs >& /dev/null

setenv NOCOMPILE 1

set get_stats = 0
if( $#argv == 1 ) then
  set get_stats = 1
endif

foreach g ( `ls ../data-sets/input//medium/*.in-offsets | grep -v sym | sed "s/.*med/med/g" | sed "s/.in-off.*//g"` `ls ../data-sets/input/*.in-offsets | grep -v sym | sed "s/.*input\///g" | sed "s/.in-off.*//g"`)

  set gname = `basename $g`

  printf "$gname "

  # if( $get_stats == 0 ) then
  #   printf "\n"
  # endif

  printf "\n"

  setenv stats_file_name ../outputs/$gname.stats_untiled_nobuffers.txt
  cd tiled-compressed 
  if( $get_stats ) then
    $PYTHON ../util/stats_parser.py $stats_file_name
  else
    echo " --> untiled-nobuffers"
    qsub -P $PROJECT -n $CPUS -q $QUEUE -oo $stats_file_name:r.lsf.out ./run.sh $g -1 -1 $stats_file_name
  endif
  cd ../
  
  setenv stats_file_name ../outputs/$gname.stats_untiled_withbuffers.txt
  cd untiled-compressed 
  if( $get_stats ) then
    $PYTHON ../util/stats_parser.py $stats_file_name
  else
    echo " --> untiled-withbuffers"
    qsub -P $PROJECT -n $CPUS -q $QUEUE -oo $stats_file_name:r.lsf.out ./run.sh $g $stats_file_name
  endif
  cd ../
 
  setenv stats_file_name ../outputs/$gname.stats_dst_tiled.txt
  cd tiled-compressed 
  if( $get_stats ) then
    $PYTHON ../util/stats_parser.py $stats_file_name
  else
    echo " --> dst tiled"
    qsub -P $PROJECT -n $CPUS -q $QUEUE -oo $stats_file_name:r.lsf.out ./run.sh $g -1 $TILE_SIZE $stats_file_name
  endif
  cd ../
 
  setenv stats_file_name ../outputs/$gname.stats_src_tiled.txt
  cd tiled-compressed 
  if( $get_stats ) then
    $PYTHON ../util/stats_parser.py $stats_file_name
  else
    echo " --> src tiled"
    qsub -P $PROJECT -n $CPUS -q $QUEUE -oo $stats_file_name:r.lsf.out ./run.sh $g $TILE_SIZE -1 $stats_file_name
  endif
  cd ../
 
  setenv stats_file_name ../outputs/$gname.stats_src_and_dst_tiled.txt
  cd tiled-compressed 
  if( $get_stats ) then
    $PYTHON ../util/stats_parser.py $stats_file_name
  else
    echo " --> src and dst tiled"
    qsub -P $PROJECT -n $CPUS -q $QUEUE -oo $stats_file_name:r.lsf.out ./run.sh $g $TILE_SIZE $TILE_SIZE $stats_file_name
  endif
  cd ../
 
    echo ""
end

