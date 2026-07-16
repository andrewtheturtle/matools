#' @import GenomicRanges
#' @import data.table
#' @import gChain
#' @import gUtils
#' @import gGnome



#' @name alignments2gw
#' @title alignments2gw
#'
#' @description
#' taken from alignments2gg, spits out intermediate gWalk object w/ nodes + edges lifted to ref coords
#' 
#' @param alignments GRanges or GRangesList of pooled reads
#' @param verbose (default = T)
#' @return gWalk object with nodes and edges lifted to reference coordinates
#' @author Marcin Imielinski, Joe DeRose, Xiaotong Yao, andrew ma
alignments2gw = function(alignments, verbose = TRUE)
{
 
  if (inherits(alignments, 'GRangesList') | inherits(alignments, 'CompressedGRangesList')){
      alignments = grl.unlist(alignments)
  }

  if (!inherits(alignments, 'GRanges') || !all(c('qname', 'cigar', 'flag') %in%  names(values(alignments))))
    stop('alignments input must be GRanges with fields $qname $cigar and $flag')

  if (verbose)
    message('making cgChain')

  cg = gChain::cgChain(alignments)

  if (verbose)
    message('disjoining query ranges and lifting nodes to reference')

  lgr = gChain::links(cg)$x
  verboten = c("seqnames", "ranges",
    "strand", "seqlevels", "seqlengths", "isCircular", "start", "end",
    "width", "element")
  values(lgr) = cbind(values(lgr), values(cg)[, setdiff(names(values(cg)), verboten)])
  
  # split the links into a GRangesList by read qname
  grb <- grbind(lgr, si2gr(gChain::links(cg)$x))
  grl <- split(grb, grb$grl.ix)
  # then lapply() gr.disjoin() on each read individually, also incorporate the qname mapping here
  grc <- lapply(grl, function(gr){
    grd <- gr.disjoin(gr)
    grd$qname <- unique(gr$qname)   # using gr$qname instead of seqnames(gr) because faster
    return(grd)
  })
  names(grc) <- unlist(lapply(grc, function(gr) unique(gr$qname)))

  gwc = gW(grl = GRangesList(grc))

  nodes = gwc$graph$nodes

  grr = gChain::lift(cg, nodes$gr)    # map to ref

  grr <- grr[order(grr$query.id)]   # just make sure in order
  grw <- gW(grl = split(grr, grr$qname))
 
  return(grw)
}



#' @name pad.walk
#' @title pad.walk
#'
#' @description
#' fills in trivial gaps between walk nodes; walks along path and can pad asymmetrically
#' 
#' @param gw input gWalk object
#' @param gap.thresh (default = 2) threshold for gap sizes to be filled
#' @return gWalk object
#' @author andrew ma
pad.walk = function(gw, gap.thresh = 2)
{
  if(!inherits(gw, 'gWalk')) stop("input must be a gWalk object")

  dt <- gr2dt(gw$grl)
  # flip (-) strands and compute gaps
  dt[, `:=`( sstart = ifelse((strand == "-"), (end), (start)),
             send   = ifelse((strand == "-"), (start), (end)) )]
  dt[, gap := abs(sstart - shift(send, 1L)), by = seqnames]

  # fill in gaps within the threshold
  dt[!is.na(gap) & gap <= gap.thresh, `:=`( nstart = ifelse((strand == "-"), (sstart + gap-1), (sstart - gap+1)) )]
  dt[is.na(nstart), nstart := sstart]

  # flip back
  dt[, `:=`( start  = ifelse((strand == "-"), (send), (nstart)),
             end    = ifelse((strand == "-"), (nstart), (send) ))]

  # (make sure not to change up row order)
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



#' @name read2node
#' @title read2node
#'
#' @description
#' maps reads GRangesList to a sequence of node id's built from the reference graph
#' 
#' @param alignments GRangesList or GRanges object of reads from BAM
#' @param gg gGraph object of reference graph that you want to map node.id from
#' @param gap (default = 2) integer for largest gap size to close
#' @param inv (defaul = 50) integer for largest inversion interval to toss
#' @param verbose (default = TRUE) logical for printing progress messages
#' @return gWalk object
#' @author andrew ma
reads2node = function(alignments = NULL, gg = NULL, gap = 2, inv = 50, verbose = TRUE)
{
  if(!inherits(alignments, 'GRangesList') && !inherits(alignments, 'GRanges')) stop("alignments must be a GRangesList or GRanges object")
  if(!inherits(gg, 'gGraph')) stop("gg must be a gGraph object")

  if(verbose) message("converting read into walk...")
  raw.gws <- alignments2gw(alignments)

  # Will eventually make this run on each read in parallel...
  ann.gws <- lapply(seq_along(raw.gws), function(i){
    tryCatch({
      raw.gw <- raw.gws[i]
      if(verbose) message(sprintf("processing read %s of %s", i, length(raw.gws)))
      
      if(verbose) message(sprintf("filling in %s bp gaps in the read walk", gap))
      pad.gw <- pad.walk(raw.gw, gap.thresh = gap)
      simp.gw <- pad.gw$copy$simplify()

      if(verbose) message(sprintf("removing small inversions under %s bp", inv))
      clean.gw <- rminv.walk(simp.gw, inv.thresh = inv)

      if(verbose) message("mapping ref gg breakpoints")
      breaks <- gg$junctions$breakpoints %&% unlist(clean.gw$grl) %>% gr.stripstrand() %>% unique()
      ann.gr <- gr.breaks.ordered(breaks, unlist(clean.gw$grl))   # bugged
      
      if(verbose) message("annotating with ref gg node.ids")
      ann.gr <- gr.val(ann.gr, gg$nodes$gr, val = "node.id", FUN = unique)     # shouldn't break so long as each gr in the walk maps to a unique node.id!
      mcols(ann.gr)$map.node.id = mcols(ann.gr)$node.id
      
      if(verbose) message("generating new walk with $map.node.id")
      ann.gw <- gW(grl = split(ann.gr, ann.gr$qname))
      ann.gw$mark( label = mcols(unlist(ann.gw$grl))$map.node.id )

      if(verbose) message(sprintf("returning walk of length %s", ann.gw$dt$length))
      return(ann.gw)

    }, error = function(e){
      
      if(verbose) message(sprintf("Error processing read %s: %s", i, e$message))
      return(NULL)

    })
  })

  ann.gr <- grbind(lapply(ann.gws,function(w){unlist(w$grl)}))
  ann.walkset <- gW(grl = split(ann.gr, ann.gr$qname))

  return(ann.walkset)
}
