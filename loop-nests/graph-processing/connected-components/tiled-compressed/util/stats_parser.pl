#!/usr/bin/perl

while( <> ) 
{
    if(/^\s+(\S+):/) {
        $metric = $1;
#        printf("Got Metric: $metric\n");
    }
    elsif(/Totals:/) {
        $metric = "Totals";
#        printf("Got Metric: $metric\n");
    }
    elsif(/Offchip reads:\s+(\d+)/)
    {
#        printf("\tStoring: $metric reads = $1\n");
        $OFFCHIP_READS{ $metric } += $1;
    }
    elsif(/Offchip updates:\s+(\d+)/)
    {
#        printf("\tStoring: $metric updates = $1\n");
        $OFFCHIP_UPDATES{ $metric } += $1;
    }
}

$dat_offchip_rd_reqs = 0;
$dat_offchip_wr_reqs = 0;
$dat_offchip_reqs = 0;
#foreach $metric( "Totals" ) 
foreach $metric ( "domain_offchip",  "indegrees_offchip", "vData_offchip" ) 
{
#     printf("LOOP: $metric $OFFCHIP_READS{$metric} $OFFCHIP_UPDATES{$metric}\n");
    $dat_offchip_reqs += $OFFCHIP_READS{$metric} + $OFFCHIP_UPDATES{$metric};
    $dat_offchip_rd_reqs += $OFFCHIP_READS{$metric};
    $dat_offchip_wr_reqs += $OFFCHIP_UPDATES{$metric};
}

$adjmat_offchip_rd_reqs = 0;
$adjmat_offchip_wr_reqs = 0;
$adjmat_offchip_reqs = 0;

foreach $metric ( "TileOffsets_offchip" , "TileSources_offchip", "inoffsets_offchip", "sources_offchip", "offset_copy_offchip", "TileVWN_offchip", "TileVWN_cnt_offchip" ) 
{
    $adjmat_offchip_reqs += $OFFCHIP_READS{$metric} + $OFFCHIP_UPDATES{$metric};
    $adjmat_offchip_rd_reqs += $OFFCHIP_READS{$metric};
    $adjmat_offchip_wr_reqs += $OFFCHIP_UPDATES{$metric};
}


if( $adjmat_offchip_reqs == 0 ) {
    printf("EXIT NODATA\n");
    exit 1;
}

printf("dat_offchip_rd_reqs $dat_offchip_rd_reqs\n");
printf("dat_offchip_wr_reqs $dat_offchip_wr_reqs\n");
printf("dat_offchip_reqs $dat_offchip_reqs\n");
printf("adjmat_offchip_rd_reqs $adjmat_offchip_rd_reqs\n");
printf("adjmat_offchip_wr_reqs $adjmat_offchip_wr_reqs\n");
printf("adjmat_offchip_reqs $adjmat_offchip_reqs\n");
printf("tot_offchip_reqs %d\n", ($adjmat_offchip_reqs + $dat_offchip_reqs));
printf("components %d\n", $components);

exit 0;
