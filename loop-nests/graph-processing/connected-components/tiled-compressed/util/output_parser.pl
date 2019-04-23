#!/usr/bin/perl

$degree = 0;

$iterations = 0;
$tile_cnt   = 0;
$tile_zero  = 0;
$tile_sum   = 0;
$all_access = 0;
$all_miss   = 0;
$done_exec  = 0;

while( <> ) 
{
    if(/Average Degree:\s+(\S+)/) {
        $degree = $1;
    }
    elsif(/Number of Edges:\s+(\d+)/) {
      $tot_edges = $1;
    }
    elsif(/NNZ: (\d+) \S+: (\d+) \S+: (\d+) \S+: (\d+) \S+: (\d+)/) {
        $tile_nnz_cnt[ $tile_cnt ] = $1;
        $tile_cnt++;
        if( $1 == 0 ) {
            $tile_zero++;
        }
        $tile_sum += $1;

        $s_access    += $2;
        $d_access    += $4;

        $s_miss      += $3;
        $d_miss      += $5;

        $all_access += $2+$4;
        $all_miss   += $3+$5;
    }
    elsif(/NNZ: (\d+)/) {
        $tile_nnz_cnt[ $tile_cnt ] = $1;
        $tile_cnt++;
        if( $1 == 0 ) {
            $tile_zero++;
        }
        $tile_sum += $1;
    }
    elsif(/TERM_RUNLIMIT/) {
        printf("EXIT RUNLIM");
        exit 1;
    }
    elsif(/TERM_MEMLIMIT/) {
        printf("EXIT MEMLIM");
        exit 1;
    }
    elsif(/Run time/) {
        $done_exec = 1;
    }
    elsif(/Exited with exit code (\d+)/)
    {
#        printf("EXIT E$1");
#        exit 1;
    }
    elsif(/Number of Vertices:\s+(\d+)/)
    {
        $vertex_cnt = $1;
    }
    elsif(/Total number of components = (\d+)/) {
        $components = $1;
    }
    elsif(/Iterations until convergence = (\d+)/) {
        $iterations = $1;
    }
}

$zero_tiles     = 0;
$avg_nnz        = 0;
$stdev_nnz      = 0;
if( $tile_cnt ) 
{
    $nnz_tiles  = ($tile_cnt-$tile_zero);
    $zero_tiles = $tile_zero / $tile_cnt * 100;
    $avg_nnz    = $tile_sum  / $nnz_tiles;

    # calculate stdev
    $i=0;
    for($t=0; $t<$tile_cnt; $t++)
    {
        
        if( $tile_nnz_cnt[$t] ) 
        {
            $tile_nnz_only[$i++] = $tile_nnz_cnt[$t];
            $stdev_nnz += ($tile_nnz_cnt[$t] - $avg_nnz)*($tile_nnz_cnt[$t] - $avg_nnz)
        }
    }

    $stdev_nnz /= $nnz_tiles;
    $stdev_nnz = sqrt($stdev_nnz);

    # sort the nnz tile
    @sorted_nnz_only = sort { $a <=> $b } @tile_nnz_only;

    $q2 = int(($nnz_tiles)/2);
    $q1 = int($q2/2);
    $q3 = int($q1*3);
    $q4 = $nnz_tiles-1;

    $e1 = int($q1/2);
    $e7 = $q3 + $e1;

#     printf("$q1,$q2,$q3,$q4\n");
#     printf("@sorted_nnz_only\n");
#    printf("Quartiles: $sorted_nnz_only[$q1],$sorted_nnz_only[$q2],$sorted_nnz_only[$q3],$sorted_nnz_only[$q4]\n");
#    printf("Quartiles: %.4f,%.4f,%.4f,%.4f\n", 100*$sorted_nnz_only[$q1]/$tot_edges,100*$sorted_nnz_only[$q2]/$tot_edges,100*$sorted_nnz_only[$q3]/$tot_edges,100*$sorted_nnz_only[$q4]/$tot_edges);

}
elsif( $degree == 0 && $tot_edges == 0)
{
    printf("EXIT NotDone");
    exit 1;
}

printf("Iterations:    $iterations\n");
printf("Components:    $components\n");
printf("Vertex:        $vertex_cnt\n");
printf("Edges:         $tot_edges\n");
printf("Degree:        %.2f\n", $degree);
printf("degratio:      %.2f\n", ($degree/$vertex_cnt*100));
printf("ZeroTilePct:   %.2f\n", $zero_tiles);
printf("AvgNNZ:        %.2f\n", $avg_nnz);
printf("StdevNNZ:      %.2f\n", $stdev_nnz);
printf("Reuse          %.2f\n", $all_miss/$all_access);
printf("AllAccess      %d\n",   $all_access);
printf("AllMiss        %d\n",   $all_miss);
printf("SAccess        %d\n",   $s_access);
printf("SMiss          %d\n",   $s_miss);
printf("DAccess        %d\n",   $d_access);
printf("DMiss          %d\n",   $d_miss);
printf("Quartiles:     $sorted_nnz_only[$q1],$sorted_nnz_only[$q2],$sorted_nnz_only[$q3],$sorted_nnz_only[$q4]\n");
printf("QuartilesPct:  %.4f,%.4f,%.4f,%.4f,%.4f\n", 100*$sorted_nnz_only[0]/$tot_edges,100*$sorted_nnz_only[$q1]/$tot_edges,100*$sorted_nnz_only[$q2]/$tot_edges,100*$sorted_nnz_only[$q3]/$tot_edges,100*$sorted_nnz_only[$q4]/$tot_edges);
#printf("E7:            %.4f\n", 100*$sorted_nnz_only[$e7]/$tot_edges);
exit 0;
