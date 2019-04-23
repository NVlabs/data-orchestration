#!/bin/tcsh
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

set OVERRIDE_TILE_SIZE = 1


source ~/mytools_workspace/whoop/functional-model/util/setup_whoop_env.csh

set    currdir = `pwd`

set EXEC    = "./cc-tiled-dst-stn.bin"

set PROJECT = research_arch_misc
set CPUS    = 1
#set QUEUE   = o_cpu_8G_24H   
set QUEUE   = o_cpu_32G_24H   

setenv NOCOMPILE 1

set get_stats = 0
set datscript = output_parser.pl

if( $#argv == 1 ) then
  set metric = $1
  set get_stats = 1
endif

if( $#argv == 2 ) then
  set metric    = $1
  set get_stats = 1
  set datscript = $2"_parser.pl"
endif

#foreach g ( ../../data-sets/input//*/*sym*.in-offsets )
#foreach g ( ../../data-sets/input//[m]*/*sym*.in-offsets )
#foreach g ( ../../data-sets/input//big/*sym*.in-offsets )
#foreach g ( ../../data-sets/input//big/*com-lj*orig*sym*.in-offsets )

foreach g ( ../../data-sets/input//big/*orig*sym*.in-offsets )
#foreach g ( ../../data-sets/input//medium/*orig*sym*.in-offsets )
  set dname    = `dirname $g`
  set bname    = `basename $dname`
  set gname    = "$bname""_"`basename $g:r`
  set graph    = "$bname""/"`basename $g:r`

  printf "$gname "
  if( $get_stats == 0 ) then
    printf "\n"
  endif

#foreach mode ( "preprocessed" "onlinetiling" )
foreach mode ( "preprocessed" )
#foreach mode ( "onlinetiling" )
  set odir = "outputs_$mode"
  mkdir $odir >& /dev/null

  if( $mode == "onlinetiling" ) then
    set EXEC    = "./cc-tiled-dst-stn-pgen.bin"
  else if( $mode == "preprocessed" ) then
    set EXEC    = "./cc-tiled-dst-stn.bin"
  endif

  foreach granularity ( 16 ) # 1 16
#  foreach tile_size ( 128 ) #262144 ) #128 256 512 1024 2048 1300 ) #262144 ) #65536 262144 ) #256 512 1024 2048 4096 262144 ) #32 256 1024 8192 ) #16384 131072 262144 )
  foreach tile_size ( 262144 ) #128 256 512 1024 2048 1300 ) #262144 ) #65536 262144 ) #256 512 1024 2048 4096 262144 ) #32 256 1024 8192 ) #16384 131072 262144 )
#  foreach tile_size ( 128 256 512 1024 2048 1300 ) #262144 ) #65536 262144 ) #256 512 1024 2048 4096 262144 ) #32 256 1024 8192 ) #16384 131072 262144 )
#  foreach dst_tile( 0 1 16 64 128 $tile_size )
  foreach dst_tile( 0 $tile_size )
  foreach src_tile( 0 $tile_size )

    set skip = 0;
    set tile_size_override = 0

    if( $dst_tile == 0 && $src_tile == 0 ) then
      set ext = "untiled"

      if( $OVERRIDE_TILE_SIZE ) then
        set ext = "untiled_override"
        set tile_size_override = 1
      endif
    else if( $dst_tile != 0 && $src_tile != 0 ) then
      set ext = "src_tile.$src_tile.dst_tile.$dst_tile"
    else if( $dst_tile == 0 ) then
      set ext = "src_tile.$src_tile"
    else  if( $src_tile == 0 && $dst_tile > 0 ) then
      set skip = 1
    else 
      set skip = 1
      set ext = "dst_tile.$dst_tile"
    endif
   
    set ext = "G.$granularity.$ext"
    
    set stats_file_name   = $odir/$gname.stats_$ext.txt
    set nbstats_file_name = $odir/$gname.stats_$ext.lsf.out

    set parsefile = $nbstats_file_name
    if( $datscript == "stats_parser.pl" ) then
      set parsefile = $stats_file_name
    endif

    set runscript=util/$datscript

    if( $skip == 0 ) then
    if( $get_stats ) then
      #cat $stats_file_name |& $runscript
      cat $parsefile |& $runscript >& /tmp/aj_stats.dat
      
      if( $status == 1 ) then
        set stat = `cat /tmp/aj_stats.dat | getcol 1`
      else
        set stat = `cat /tmp/aj_stats.dat | grep -i $metric | getcol 1`
      endif
      printf "$stat "
    else
      if( -r $nbstats_file_name ) then
        printf " --> $ext --  NOT RUNNING\n"
      else
        printf " --> $ext -- RUNNING "

        set INPUT_GRAPH   = "--vec_inoffsets_file=../../data-sets/input//$graph.in-offsets   --vec_sources_file=../../data-sets/input//$graph.sources"
        set OUTPUT_FILE   = "--vec_domain_file=/tmp/output.`basename $graph`.tiled-dst-stn.compressed.file"
        set PARAMS        = "--src_tile_size=$src_tile   --dst_tile_size=$dst_tile   --stats=$stats_file_name   --granularity=$granularity   --trace_level=0"

        set OVERRIDE_ARGS = ""
        if( $tile_size_override == 1 ) then
          set OVERRIDE_ARGS = "$OVERRIDE_ARGS --override_tile_size=$tile_size"
        endif


        set CMD           = "setenv WHOOP_CHECK_REFERENCE 0 ; $EXEC $INPUT_GRAPH $OUTPUT_FILE $PARAMS $OVERRIDE_ARGS"

#        echo "qsub -P $PROJECT -n $CPUS -q $QUEUE -oo $nbstats_file_name $CMD"
        qsub -P $PROJECT -n $CPUS -q $QUEUE -oo $nbstats_file_name $CMD
      endif
    endif # -- get stats
    endif # -- skip
  end
  end
  end
  end
  end

  if( $get_stats == 1 ) then
    echo ""
  endif
end
