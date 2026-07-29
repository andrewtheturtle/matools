#' @import data.table
#' @import GenomicRanges
#' @import GenomcAlignments
#' @import gChain
#' @import gUtils
#' @import gGnome



#' @name smooth.cigar
#' @title smooth.cigar
#' 
#' @description 
#' makes a smoothed cigar by removing I's and converting D's to M's, all w/in a certain width threshold
#' 
#' @param alignments (default = NULL) GRanges or GRangesList of pooled reads
#' @param smooth.thresh (default = 50) integer for largest deletion size to smooth
#' @return GRanges or GRangesList of pooled reads with smoothed cigar strings
#' @author andrew ma
smooth.cigar = function(alignments = NULL, smooth.thresh = 50)
{
  if (inherits(alignments, 'GRangesList') | inherits(alignments, 'CompressedGRangesList')){
      alignments = grl.unlist(alignments)
  }

  if (!inherits(alignments, 'GRanges') || !all(c('qname', 'cigar', 'flag') %in%  names(values(alignments))))
    stop('alignments input must be GRanges with fields $qname $cigar and $flag')

  # reads to cigars
  cigars.dt <- mcols(alignments) %>% as.data.table() %>% .[, .(qname, cigar, flag)]

  # cigar editing: merge I's and D's into M's if they are below a certain threshold; preserves cigar query lens and qwidth
  ops <- data.table(
    listid = dunlist(explodeCigarOps(cigars.dt$cigar))$listid,
    c.str  = as.character(dunlist(explodeCigarOps(cigars.dt$cigar))$V1),
    c.len  = as.integer(dunlist(explodeCigarOpLengths(cigars.dt$cigar))$V1))

  ops[, mergeable := c.str %in% c("M","=","X") |
                    (c.str %in% c("I","D") & c.len <= smooth.thresh)]
  ops[, qlen := fifelse(c.str %in% c("M","=","X","I"), c.len, 0L)]
  ops[, run  := rleid(mergeable), by = listid]

  new <- ops[, if (mergeable[1L]) .(c.str = "M", c.len = sum(qlen))
              else               .(c.str = c.str, c.len = c.len),
            by = .(listid, run)][c.len > 0L]

  agg <- new[, .(cigar = paste0(c.len, c.str, collapse = "")), by = listid]
  new.cigar <- agg$cigar[match(seq_len(nrow(cigars.dt)), agg$listid)]
  
  stopifnot(!anyNA(new.cigar))
  stopifnot(all(cigarWidthAlongQuerySpace(new.cigar) == mcols(alignments)$qwidth))

  # re-map back to alignments
  mcols(alignments)$cigar <- new.cigar

  return(alignments)
}



#' @name alignments2gw
#' @title alignments2gw
#'
#' @description
#' taken from alignments2gg, spits out intermediate gWalk object w/ nodes + edges lifted to ref coords
#' 
#' @param alignments GRanges or GRangesList of pooled reads
#' @param ignore.overlaps (default = T) logical for whether to ignore overlaps in alignments when lifting to reference
#' @param drop (default = 0) integer for dropping nodes in read space
#' @param verbose (default = T)
#' @return gWalk object with nodes and edges lifted to reference coordinates
#' @author Marcin Imielinski, Joe DeRose, Xiaotong Yao, andrew ma
alignments2gw = function(alignments, drop = 0, ignore.overlaps = FALSE, verbose = TRUE)
{
 
  if (inherits(alignments, 'GRangesList') | inherits(alignments, 'CompressedGRangesList')){
      alignments = grl.unlist(alignments)
  }

  if (!inherits(alignments, 'GRanges') || !all(c('qname', 'cigar', 'flag') %in%  names(values(alignments))))
    stop('alignments input must be GRanges with fields $qname $cigar and $flag')

  if (verbose)
    message('making cgChain')

  cg = gChain::cgChain(alignments)

  lgr = gChain::links(cg)$x
  verboten = c("seqnames", "ranges",
    "strand", "seqlevels", "seqlengths", "isCircular", "start", "end",
    "width", "element")
  values(lgr) = cbind(values(lgr), values(cg)[, setdiff(names(values(cg)), verboten)])
  
  # split the links into a GRangesList by read qname
  grl <- split(lgr, lgr$qname)
  # then lapply() gr.disjoin() on each read individually, also incorporate the qname mapping here
  grc <- lapply(grl, function(gr){
    grd <- gr.disjoin(gr)

    # toss nodes in read space
    grd <- grd[width(grd) > drop]

    grd$qname <- unique(gr$qname)   # using gr$qname instead of seqnames(gr) because faster
    return(grd)
  })
  grc <- grl.unlist(GRangesList(grc))   # unlist the GRangesList back into a single GRanges object
  grr = gChain::lift(cg, grc)    # map to ref

  # disjoin and lift will create new nodes for overlapping alignment records
  # however, we don't have an automatable heuristic for mapping the overlap to the correct segment based on basepairs
  # instead, we'll use original link.ids to collapse overlapping segments back to the original link ranges
  if (ignore.overlaps)
  {
    if (verbose)
      message('>>> mapping back to original links to collapse overlapping segments...')

    mcols(grr)$links.x.ranges <- ranges(gChain::links(cg)$x[mcols(grr)$link.id])
    mcols(grr)$links.y.ranges <- gChain::links(cg)$y[mcols(grr)$link.id]
    
    # create new grr with original link ranges
    new.grr <- GRanges(seqnames = seqnames(grr), ranges = ranges(mcols(grr)$links.y.ranges), strand = strand(grr))
    
    # grr already has the ranges correctly ordered by original link ranges
    # unique() will collapse to the first instance of each link.id, which should preserve order
    # grl and query information don't seem to be ordered
    remove_cols <- c("grl.ix","grl.iix","query.id","query.start","query.end")
    mcols(new.grr) <- mcols(grr)[, setdiff(names(mcols(grr)), remove_cols)]
    # if an overlapping segment on the link does not actually align to the reference and we get some bleeding into the next node,
    # that will be accounted for via junction-based walk constructions downstream (ra.overlaps with a pad)
    grr <- unique(new.grr)
  }
    
  grw <- gW(grl = split(grr, grr$qname))
 
  return(grw)
}



#' @name pad.walks
#' @title pad.walks
#'
#' @description
#' fills in trivial gaps between nodes for each walk; walks along path and can pad asymmetrically
#' 
#' @param gw input gWalk object
#' @param gap.thresh (default = 2) threshold for gap sizes to be filled
#' @return gWalk object
#' @author andrew ma
pad.walks = function(gw, gap.thresh = 2)
{
  if(!inherits(gw, 'gWalk')) stop("input must be a gWalk object")

  dt <- gr2dt(gw$grl)
  # flip (-) strands
  dt[, `:=`( sstart = ifelse((strand == "-"), (end), (start)),
             send   = ifelse((strand == "-"), (start), (end)) )]
  
  # compute gap per walk (qname) > per chr
  dt[, gap := abs(sstart - shift(send, 1L)), by = c("qname", "seqnames")]

  # fill in gaps within the threshold
  dt[!is.na(gap) & gap <= gap.thresh, `:=`( nstart = ifelse((strand == "-"), (sstart + gap-1), (sstart - gap+1)) )]
  dt[is.na(nstart), nstart := sstart]

  # flip back
  dt[, `:=`( start  = ifelse((strand == "-"), (send), (nstart)),
             end    = ifelse((strand == "-"), (nstart), (send) ))]

  gr <- dt2gr(dt)
  new.gw <- gW(grl = split(gr, gr$qname))

  return(new.gw)
}



#' @name rminv.walk
#' @title rminv.walk
#'
#' @description
#' remove trivial inversions under a size threshold for walks
#' 
#' @param gw input gWalk object
#' @param inv.thresh (default = 1) threshold for inversions to be dropped
#' @return gWalk object
#' @author andrew ma
rminv.walk = function(gw, inv.thresh = 1)
{
  if(!inherits(gw, 'gWalk')) stop("input must be a gWalk object")

  dt <- gr2dt(gw$grl)
  
  # identify strand conversions
  dt[, strand.change := ifelse( (strand==shift(strand,1L)),(FALSE),(TRUE) )]

  # drop the smol inversions
  drop.id <- dt[strand.change & width<=inv.thresh,node.id]
  if(length(drop.id) > 0) {
    new.dt <- dt[-c(drop.id)]
  } else {
    new.dt <- dt
  }
  new.gr <- dt2gr(new.dt)

  new.gw <- gW(grl = split(new.gr, new.gr$qname))

  return(new.gw)
}



#' @name gr.breaks.ordered
#' @title gr.breaks.ordered
#' @description
#'
#' Break GRanges at given breakpoints into disjoint gr
#' edit: returns GRanges in original order
#'
#' @author Xiaotong Yao, andrew ma
#' @param bps GRanges of width 1, locations of the breakpoints; if any element width
#' larger than 1, both boundary will be considered individual breakpoints
#' @param query a disjoint GRanges object to be broken
#' @return Granges disjoint object at least the same length as query,
#' with metadata col `qid` indicating input index where new segment is from and
#' `node_ord` indicating order of new resultant segments
gr.breaks.ordered = function(bps=NULL, query=NULL)
{
   ## ALERT: big change! input parameter shuffled!
   ## if bps not provided, return back-traced disjoin wrapper
   if (is.null(bps)) {
       message("Argument 'bps' not provided")
       return(query)
   } else {
       ## only when bps is given do we care about what query is
       if (is.null(query)){
           query = gr.stripstrand(si2gr(seqinfo(bps)))
       }

       ## in case query is not a GRanges
       if (!is(query, "GRanges")){
           stop("Error: 'query' must be a GRanges object.")
       }

       query$qid = seq_along(query)

       ## preprocess bps
       ## having meta fields? remove them!
       bps = bps[, c()]

       ## remove things outside of ref
       oo.seqlength = which(start(bps)<1 | end(bps)>seqlengths(bps)[as.character(seqnames(bps))])
       
       if (length(oo.seqlength)>0){
           warning("Warning: Some breakpoints out of chr lengths. Removing.")
           bps = bps[-oo.seqlength]
       }

       if (any(!is.null(names(bps)))){
           warning("Warning: Removing row names from bps.")
           names(bps) = NULL
       }

       ## having strand info? remove it!
       if (any(strand(bps)!="*")){
           warning("Warning: Some breakpoints have strand info. Force to '*'.")
           bps = gr.stripstrand(bps)
       }

       ## solve three edge cases
       if (any(w.0 <- (width(bps)<1))){
           warning("Warning: Some breakpoint width==0. Discard.")
           bps = bps[-which(w.0)]
       }

       if (any(w.2 <- (width(bps)==2))){
           warning("Warning: Some breakpoint width>2. Will tear them apart and treat as two breakpoints.")
           ## this is seen as breakpoint by spanning two bases
           bps = c(bps[-which(w.2)],
                   gr.start(bps[which(w.2)]),
                   gr.end(bps[which(w.2)]))
       }

       if (any(w.l <- (width(bps)>2))){
           ## some not a point? turn it into a point
           warning("Warning: Some breakpoint width>2. Treat them as segmentations.")
           rbps = gr.end(bps[which(w.l)])
           lbps = gr.start(bps[which(w.l)])
           start(lbps) = pmax(start(lbps)-1, 1)
           bps = c(bps[which(!w.l)], streduce(c(lbps, rbps)))
       }

       bps$inQuery = bps %^% query
       if (any(bps$inQuery==FALSE)){
           warning("Warning: Some breakpoint not within query ranges.")
       }

       ## label and only consider breakpoints not already at the boundary of query
       bps$inner = bps$inQuery ## out of query automatically FALSE
       bps$inner[which(bps %^% gr.start(query) | bps %^% gr.end(query))]=FALSE
       
       ## maybe no inner bp at all, then no need to proceed
       if (!any(bps$inner)){
           return(query)
       }
       bpsInner = bps %Q% (inner==T)

       ## map query and inner breakpoints
       qbMap = gr.findoverlaps(query, bpsInner)
       mappedQ = seq_along(query) %in% qbMap$query.id
       ## raw coors to construct ranges from
       tmpRange = data.table(qid2 = qbMap$query.id,
                             startFrom = start(query[qbMap$query.id]),
                             breakAt = start(bpsInner[qbMap$subject.id]),
                             upTo = end(query[qbMap$query.id]))
       tmpCoor = tmpRange[, .(pos=sort(unique(c(startFrom, breakAt, upTo)))), by=qid2]

       ## construct new ranges
       newRange = tmpCoor[, .(tmp.start=pos[-which.max(pos)],
                              end=pos[-which.min(pos)]),
                          by=qid2]
       newRange[, ":="(seqnames = as.vector(seqnames(query)[qid2]),
                       strand = as.vector(strand(query)[qid2]))]
       newRange[, start := ifelse(tmp.start==min(tmp.start), tmp.start, tmp.start+1), by=qid2]

       ## strand-aware traversal ordinal: + ascending, - reversed
       newRange[, node_ord := if (strand[1] == "-") rev(seq_len(.N)) else seq_len(.N), by=qid2]

       ## put together the mapped and broken
       newGr = GRanges(newRange, seqinfo = seqinfo(query))
       values(newGr) = values(query)[newGr$qid2, , drop=F]   ## preserve the input metacol
       newGr$node_ord = newRange$node_ord                    ## re-attach after the overwrite

       intact = query[!mappedQ]
       if (length(intact) > 0) {
         intact$node_ord = 1L                                  ## unbroken -> single piece (sometimes returns empty)
       }

       output = c(newGr, intact)
       output = output[order(output$qid, output$node_ord)]   ## input order; traversal order within
       return(output)
   }
}



#' @name reads2node
#' @title reads2node
#'
#' @description
#' maps reads GRangesList to a sequence of node id's built from the reference graph
#' 
#' @param alignments GRangesList or GRanges object of reads from BAM
#' @param gg gGraph object of reference graph that you want to map node.id from
#' @param gap (default = 50) integer for largest gap size to close
#' @param verbose (default = TRUE) logical for printing progress messages
#' @return gWalk object
#' @author andrew ma
reads2node = function(alignments = NULL, gg = NULL, gap = 50, verbose = TRUE)
{
  if(!inherits(alignments, 'GRangesList') && !inherits(alignments, 'GRanges')) stop("alignments must be a GRangesList or GRanges object")
  if(!inherits(gg, 'gGraph')) stop("gg must be a gGraph object")

  if(verbose) message("converting reads > walks...")
  raw.gws <- alignments2gw(alignments)

  if(verbose) message(sprintf("...filling in %s bp gaps in the read walk", gap))
  pad.gws <- pad.walks(raw.gws, gap.thresh = gap)

  if(verbose) message("...annotating with ref gg node.ids")
  ann.gr <- gr.val(grl.unlist(pad.gws$grl), gg$nodes$gr, val = "node.id", FUN = unique)
  mcols(ann.gr)$map.node.id = mcols(ann.gr)$node.id                                         # not necessary but if you want to label with diff node id's
  mcols(ann.gr)$map.snode.id = sign(mcols(ann.gr)$snode.id) * mcols(ann.gr)$map.node.id     #
  ann.gws <- gW(grl = split(ann.gr, ann.gr$qname))
  
  if(verbose) message("...simplifying")
  simp.gws <- ann.gws$copy$simplify(by = "map.node.id")   # node ids get mismapped during reduction

  return(simp.gws)
}



#' @name drop.nodes.walk
#' @title drop.nodes.walk
#'
#' @description
#' takes in a gWalk object and drops nodes that are below a specified width threshold
#' 
#' @param gws gWalk object
#' @param width.thresh (default = 50) integer for minimum node width to keep
#' @return gWalk object
#' @author andrew ma
drop.nodes.walk <- function(gws = NULL, width.thresh = 50)
{
  if(!inherits(gws, 'gWalk')) stop("gws must be a gWalk object")

  gws.gr <- grl.unlist(gws$grl)
  mcols(gws.gr)$width <- width(gws.gr)
  
  new.gws.gr <- gws.gr[mcols(gws.gr)$width > width.thresh]
  mcols(new.gws.gr)$width <- NULL
  
  new.gws <- gW(grl = split(new.gws.gr, new.gws.gr$qname))
  
  return(new.gws)
}



#' @name alignments2gg.d
#' @title alignments2gg.d
#'
#' @description
#' alignments2gg but can drop nodes below a certain width threshold at gWalk level
#' 
#' @param tile GRanges of tiles
#' @param juncs Junction object or grl coercible to Junctions object
#' @param genome seqinfo or seqlengths
#' @return list with gr and edges which can be input into standard gGnome constructor
#' @author Marcin Imielinski, Joe DeRose, Xiaotong Yao, andrew ma
alignments2gg.d = function(alignment, width.thresh = 50, verbose = TRUE)
{

  if (inherits(alignment, 'GRangesList') | inherits(alignment, 'CompressedGRangesList')){
      alignment = grl.unlist(alignment)
  }
  if (!inherits(alignment, 'GRanges') || !all(c('qname', 'cigar', 'flag') %in%  names(values(alignment))))
    stop('alignment input must be GRanges with fields $qname $cigar and $flag')

  if (verbose)
    message('making cgChain')

  cg = gChain::cgChain(alignment)

  if (verbose)
    message('disjoining query ranges and lifting nodes to reference')

  lgr = gChain::links(cg)$x
  verboten = c("seqnames", "ranges",
    "strand", "seqlevels", "seqlengths", "isCircular", "start", "end",
    "width", "element")
  values(lgr) = cbind(values(lgr), values(cg)[, setdiff(names(values(cg)), verboten)])
  grc = gr.disjoin(grbind(lgr, si2gr(gChain::links(cg)$x)))
  grc$qname = seqnames(grc)
  gwc = gW(grl = split(grc, seqnames(grc)))

  ## lol let's just try this...
  gwc <- drop.nodes.walk(gwc, width.thresh = width.thresh)

  nodes = gwc$graph$nodes
  grr = gChain::lift(cg, nodes$gr)
  grr$insertion = FALSE

  ## add a pad either to the right or left (basically, there should always be mapped sequence on one side on an insertion ..
  ## otherwise there is no alignment (ie pure insertions means no alignment)
  ix = setdiff(nodes$gr$node.id, grr$node.id)
  if (length(ix))
  {
    insertions = nodes[ix]$gr
    start(insertions) = ifelse(start(insertions)>1, start(insertions)-1, start(insertions))
    end(insertions) = ifelse(start(insertions)== 1 & end(insertions) < seqlengths(insertions)[as.character(seqnames(insertions))],
                             end(insertions)+1, end(insertions))
    insertions$insertion = TRUE
    grr = grbind(grr, gChain::lift(cg, insertions)) ## add the lifted insertions to the pile of intervals
  }

  ugrr = unique(gr.stripstrand(grr))

  ## there may be dups here if say the lift aligns the contig to both the negative and positive side
  ## of a contig
  grr$ugrr.id = match(gr.stripstrand(grr), ugrr)
  grr$grr.id = 1:length(grr)

  if (any(ugrr$insertion))
  {
    width(ugrr[ugrr$insertion]) = 0
  }

  ## find insertions ie nodes that did not survive the lift
  edges = gwc$graph$edges$dt

  if (verbose)
    message('lifting edges to reference')


  ## to lift to genome cordinates, merge old edges with new ids (will duplicate edges across multimaps)
  edges.new = edges %>% merge(gr2dt(grr), by.x = 'n1', by.y = 'node.id', allow.cartesian = TRUE) %>% merge( gr2dt(grr), by.x = 'n2', by.y = 'node.id', allow.cartesian = TRUE)
  edges.new[, n1 := ugrr.id.x] ## we map to the 
  edges.new[, n2 := ugrr.id.y]

  ## flip sides for nodes that are flipped (i.e. negative strand) during lift
  .flip = function(x) c(left = 'right', right = 'left')[x]

  edges.new$n1.side = ifelse(strand(grr)[edges.new$grr.id.x] == '-', .flip(edges.new$n1.side), edges.new$n1.side)
  edges.new$n2.side = ifelse(strand(grr)[edges.new$grr.id.y] == '-', .flip(edges.new$n2.side), edges.new$n2.side)

  ## now just need to replace any edges to and from an insertion
  ugrr$loose.left = ugrr$loose.right = NULL

  if (verbose)
    message('building graph')
  
  return(list(nodes = ugrr, edges = edges.new[, .(n1, n1.side, n2, n2.side)]))
}



#' @name get_readL
#' @title get_readL
#'
#' @description
#' outputs read length distribution and mean read length from reads (just uses $qwidth param)
#' 
#' @param reads GRanges or GRangesList of reads
#' @return list with read length distribution and mean read length
#' @author andrew ma
get_readL <- function(reads)
{
  if(inherits(reads,'GRangesList') | inherits(reads, 'CompressedGRangesList')){
    reads.gr <- grl.unlist(reads)
  }
  else if(inherits(reads, 'GRanges')){
    reads.gr <- reads
  }
  else{
    stop("reads must be a GRanges or GRangesList object")
  }

  # Get read length distribution and average read length
  u.reads.gr <- reads.gr[!is.na(mcols(reads.gr)$qwidth)]      # first remove NA qwidth entries
  u.reads.gr <- u.reads.gr[!duplicated(mcols(u.reads.gr)$qname)]
  read.lengths <- mcols(u.reads.gr)$qwidth
  mean.read.length <- mean(read.lengths)

  return(list(read.lengths = read.lengths, mean.read.length = mean.read.length))
}



#' @name snap2bps
#' @title snap2bps
#' 
#' @description
#' snaps a GRanges/GRangesList to a set of breakpoints (bps) within a supplied threshold (pad)
#' 
#' @param gr GRanges or GRangesList to be snapped
#' @param bps GRanges of breakpoints to snap to
#' @param pad (default = 5) integer for snapping threshold
#' @return GRanges or GRangesList of snapped ranges
#' @author andrew ma
snap2bps <- function(gr, bps, pad = 5)
{
  if(inherits(gr,'GRangesList') | inherits(gr, 'CompressedGRangesList')){
    gr <- grl.unlist(gr)
  }
  else if(!inherits(gr, 'GRanges')){
    stop("gr must be a GRanges or GRangesList object")
  }

  if(!inherits(bps, 'GRanges')){
    stop("bps must be a GRanges object")
  }

  # make sure breakpoints are strandless and unique, then pad
  bps <- bps %>% gr.stripstrand() %>% unique()
  bpgr <- bps + pad
  
  lgr <- gr.start(gr, ignore.strand = TRUE)     # rightmost base of segments
  rgr <- gr.end(gr, ignore.strand = TRUE)       # leftmost base of segments

  lov <- gr.findoverlaps(lgr, bpgr, ignore.strand = TRUE)
  mcols(lov)$type <- "left"
  rov <- gr.findoverlaps(rgr, bpgr, ignore.strand = TRUE)
  mcols(rov)$type <- "right"

  # combine overlaps and snap to breakpoints
  ov <- grbind(lov, rov) %>% gr2dt()
  ov[, bp := start(bps)[subject.id]]
  ov[, bpdist := start - bp]

  snap <- ov[bpdist != 0]
  gdt <- gr2dt(gr)
  gdt[, idx := .I]
  gdt[snap[type == "left"], on = .(idx == query.id), start := i.bp]
  gdt[snap[type == "right"], on = .(idx == query.id), end := i.bp]

  new.gr <- dt2gr(gdt)

  return(new.gr)
}



#' @name edgefix
#' @title edgefix
#' 
#' @description
#' fills in missing edge CN's based on a ref gg's node.id's, then loosefixes output before returning
#' ! This is used in the context of fixing up a new disjoined graph from the two parents, e.g. gg <- ref.gg$disjoin(nodes$gr)
#' 
#' @param gg gGraph object to paste edge CN's onto
#' @param ref.gg gGraph object to paste edge CN's from
#' @return gGraph object with pasted edge CN's
#' @author andrew ma
edgefix <- function(gg, ref.gg)
{
  if(!inherits(gg, 'gGraph')) stop("gg must be a gGraph object")
  if(!inherits(ref.gg, 'gGraph')) stop("ref.gg must be a gGraph object")

  fix.edges <- gg$edges[is.na(cn)]$dt$edge.id
  fix.nodes <- gg$edges[fix.edges]$dt$n1        # taking cn from origin node of edge
  cn.edges <- gg$nodes[fix.nodes]$dt$cn

  new.gg <- gg$copy
  new.gg$edges[fix.edges]$mark(cn = cn.edges)
  new.gg <- loosefix(new.gg)

  return(new.gg)
}



#' @name get.freeze
#' @title get.freeze
#' 
#' @description
#' Gets nodes with only REF edges to be frozen in sample.gwalks; returns a vector of the node id's
#' 
#' @param gg gGraph object to get frozen nodes from
#' @return vector of node id's to be frozen
#' @author andrew ma
get.freeze <- function(gg)
{
  if(!inherits(gg, 'gGraph')) stop("gg must be a gGraph object")

  samp.nodes <- as.vector(as.matrix(gg$edges$dt[gg$edges$dt$type=="ALT",.(n1,n2)]))
  freeze.nodes <- setdiff(gg$nodes$dt$node.id,samp.nodes)

  return(freeze.nodes)
}
