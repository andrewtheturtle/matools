#' @import data.table
#' @import GenomicRanges
#'@importFrom GenomicAlignments explodeCigarOps explodeCigarOpLengths cigarWidthAlongQuerySpace
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
#' @export
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
  new.alignments <- copy(alignments)
  mcols(new.alignments)$cigar <- new.cigar

  return(new.alignments)
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
#' @export
alignments2gw = function(alignments, drop = 0, ignore.overlaps = TRUE, verbose = TRUE)
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
  
  grl <- split(lgr, lgr$qname)
  # gr.disjoin() on each individual read
  grc <- lapply(grl, function(gr){
    grd <- gr.disjoin(gr)
    
    ## option to toss small nodes in read space ##
    grd <- grd[width(grd) > drop]

    grd$qname <- unique(gr$qname)
    return(grd)
  })
  grc <- grl.unlist(GRangesList(grc))
  grr = gChain::lift(cg, grc)    # map to ref

  # disjoin and lift will create new duplicate nodes for overlapping alignment records, but only one of the two actually aligns
  # instead, we'll use original link.ids to collapse overlapping segments back to the original link ranges
  if (ignore.overlaps)
  {
    if (verbose)
      message('>>> mapping back to original links to collapse overlapping segments...')

    mcols(grr)$links.x.ranges <- ranges(gChain::links(cg)$x[mcols(grr)$link.id])
    mcols(grr)$links.y.ranges <- gChain::links(cg)$y[mcols(grr)$link.id]
    
    # create new grr with original link ranges
    new.grr <- GRanges(seqnames = seqnames(grr), ranges = ranges(mcols(grr)$links.y.ranges), strand = strand(grr))
    
    # grl and query information don't seem to be ordered/helpful here
    remove_cols <- c("grl.ix","grl.iix","query.id","query.start","query.end")
    mcols(new.grr) <- mcols(grr)[, setdiff(names(mcols(grr)), remove_cols)]

    # dedup on $link.id to avoid collapsing tandem dups or inversions
    new.grr.dt <- gr2dt(new.grr)
    new.grr.dt <- new.grr.dt[!duplicated(new.grr.dt$link.id)]
    new.grr.dt <- new.grr.dt[order(new.grr.dt$qname, new.grr.dt$links.x.ranges.start)]    # ensure order is preserved
    grr <-  dt2gr(new.grr.dt)
  }
    
  grw <- gW(grl = split(grr, grr$qname))
 
  return(grw)
}



# #' @name pad.walks
# #' @title pad.walks
# #'
# #' @description
# #' fills in trivial gaps between nodes for each walk; walks along path and can pad asymmetrically
# #' 
# #' @param gw input gWalk object
# #' @param gap.thresh (default = 2) threshold for gap sizes to be filled
# #' @return gWalk object
# #' @author andrew ma
# pad.walks = function(gw, gap.thresh = 2)
# {
#   if(!inherits(gw, 'gWalk')) stop("input must be a gWalk object")

#   dt <- gr2dt(gw$grl)
#   # flip (-) strands
#   dt[, `:=`( sstart = ifelse((strand == "-"), (end), (start)),
#              send   = ifelse((strand == "-"), (start), (end)) )]
  
#   # compute gap per walk (qname) > per chr
#   dt[, gap := abs(sstart - shift(send, 1L)), by = c("qname", "seqnames")]

#   # fill in gaps within the threshold
#   dt[!is.na(gap) & gap <= gap.thresh, `:=`( nstart = ifelse((strand == "-"), (sstart + gap-1), (sstart - gap+1)) )]
#   dt[is.na(nstart), nstart := sstart]

#   # flip back
#   dt[, `:=`( start  = ifelse((strand == "-"), (send), (nstart)),
#              end    = ifelse((strand == "-"), (nstart), (send) ))]

#   gr <- dt2gr(dt)
#   new.gw <- gW(grl = split(gr, gr$qname))

#   return(new.gw)
# }



# #' @name rminv.walk
# #' @title rminv.walk
# #'
# #' @description
# #' remove trivial inversions under a size threshold for walks
# #' 
# #' @param gw input gWalk object
# #' @param inv.thresh (default = 1) threshold for inversions to be dropped
# #' @return gWalk object
# #' @author andrew ma
# rminv.walk = function(gw, inv.thresh = 1)
# {
#   if(!inherits(gw, 'gWalk')) stop("input must be a gWalk object")

#   dt <- gr2dt(gw$grl)
  
#   # identify strand conversions
#   dt[, strand.change := ifelse( (strand==shift(strand,1L)),(FALSE),(TRUE) )]

#   # drop the smol inversions
#   drop.id <- dt[strand.change & width<=inv.thresh,node.id]
#   if(length(drop.id) > 0) {
#     new.dt <- dt[-c(drop.id)]
#   } else {
#     new.dt <- dt
#   }
#   new.gr <- dt2gr(new.dt)

#   new.gw <- gW(grl = split(new.gr, new.gr$qname))

#   return(new.gw)
# }



# #' @name gr.breaks.ordered
# #' @title gr.breaks.ordered
# #' @description
# #'
# #' Break GRanges at given breakpoints into disjoint gr
# #' edit: returns GRanges in original order
# #'
# #' @author Xiaotong Yao, andrew ma
# #' @param bps GRanges of width 1, locations of the breakpoints; if any element width
# #' larger than 1, both boundary will be considered individual breakpoints
# #' @param query a disjoint GRanges object to be broken
# #' @return Granges disjoint object at least the same length as query,
# #' with metadata col `qid` indicating input index where new segment is from and
# #' `node_ord` indicating order of new resultant segments
# gr.breaks.ordered = function(bps=NULL, query=NULL)
# {
#    ## ALERT: big change! input parameter shuffled!
#    ## if bps not provided, return back-traced disjoin wrapper
#    if (is.null(bps)) {
#        message("Argument 'bps' not provided")
#        return(query)
#    } else {
#        ## only when bps is given do we care about what query is
#        if (is.null(query)){
#            query = gr.stripstrand(si2gr(seqinfo(bps)))
#        }

#        ## in case query is not a GRanges
#        if (!is(query, "GRanges")){
#            stop("Error: 'query' must be a GRanges object.")
#        }

#        query$qid = seq_along(query)

#        ## preprocess bps
#        ## having meta fields? remove them!
#        bps = bps[, c()]

#        ## remove things outside of ref
#        oo.seqlength = which(start(bps)<1 | end(bps)>seqlengths(bps)[as.character(seqnames(bps))])
       
#        if (length(oo.seqlength)>0){
#            warning("Warning: Some breakpoints out of chr lengths. Removing.")
#            bps = bps[-oo.seqlength]
#        }

#        if (any(!is.null(names(bps)))){
#            warning("Warning: Removing row names from bps.")
#            names(bps) = NULL
#        }

#        ## having strand info? remove it!
#        if (any(strand(bps)!="*")){
#            warning("Warning: Some breakpoints have strand info. Force to '*'.")
#            bps = gr.stripstrand(bps)
#        }

#        ## solve three edge cases
#        if (any(w.0 <- (width(bps)<1))){
#            warning("Warning: Some breakpoint width==0. Discard.")
#            bps = bps[-which(w.0)]
#        }

#        if (any(w.2 <- (width(bps)==2))){
#            warning("Warning: Some breakpoint width>2. Will tear them apart and treat as two breakpoints.")
#            ## this is seen as breakpoint by spanning two bases
#            bps = c(bps[-which(w.2)],
#                    gr.start(bps[which(w.2)]),
#                    gr.end(bps[which(w.2)]))
#        }

#        if (any(w.l <- (width(bps)>2))){
#            ## some not a point? turn it into a point
#            warning("Warning: Some breakpoint width>2. Treat them as segmentations.")
#            rbps = gr.end(bps[which(w.l)])
#            lbps = gr.start(bps[which(w.l)])
#            start(lbps) = pmax(start(lbps)-1, 1)
#            bps = c(bps[which(!w.l)], streduce(c(lbps, rbps)))
#        }

#        bps$inQuery = bps %^% query
#        if (any(bps$inQuery==FALSE)){
#            warning("Warning: Some breakpoint not within query ranges.")
#        }

#        ## label and only consider breakpoints not already at the boundary of query
#        bps$inner = bps$inQuery ## out of query automatically FALSE
#        bps$inner[which(bps %^% gr.start(query) | bps %^% gr.end(query))]=FALSE
       
#        ## maybe no inner bp at all, then no need to proceed
#        if (!any(bps$inner)){
#            return(query)
#        }
#        bpsInner = bps %Q% (inner==T)

#        ## map query and inner breakpoints
#        qbMap = gr.findoverlaps(query, bpsInner)
#        mappedQ = seq_along(query) %in% qbMap$query.id
#        ## raw coors to construct ranges from
#        tmpRange = data.table(qid2 = qbMap$query.id,
#                              startFrom = start(query[qbMap$query.id]),
#                              breakAt = start(bpsInner[qbMap$subject.id]),
#                              upTo = end(query[qbMap$query.id]))
#        tmpCoor = tmpRange[, .(pos=sort(unique(c(startFrom, breakAt, upTo)))), by=qid2]

#        ## construct new ranges
#        newRange = tmpCoor[, .(tmp.start=pos[-which.max(pos)],
#                               end=pos[-which.min(pos)]),
#                           by=qid2]
#        newRange[, ":="(seqnames = as.vector(seqnames(query)[qid2]),
#                        strand = as.vector(strand(query)[qid2]))]
#        newRange[, start := ifelse(tmp.start==min(tmp.start), tmp.start, tmp.start+1), by=qid2]

#        ## strand-aware traversal ordinal: + ascending, - reversed
#        newRange[, node_ord := if (strand[1] == "-") rev(seq_len(.N)) else seq_len(.N), by=qid2]

#        ## put together the mapped and broken
#        newGr = GRanges(newRange, seqinfo = seqinfo(query))
#        values(newGr) = values(query)[newGr$qid2, , drop=F]   ## preserve the input metacol
#        newGr$node_ord = newRange$node_ord                    ## re-attach after the overwrite

#        intact = query[!mappedQ]
#        if (length(intact) > 0) {
#          intact$node_ord = 1L                                  ## unbroken -> single piece (sometimes returns empty)
#        }

#        output = c(newGr, intact)
#        output = output[order(output$qid, output$node_ord)]   ## input order; traversal order within
#        return(output)
#    }
# }



# #' @name reads2node
# #' @title reads2node
# #'
# #' @description
# #' maps reads GRangesList to a sequence of node id's built from the reference graph
# #' 
# #' @param alignments GRangesList or GRanges object of reads from BAM
# #' @param gg gGraph object of reference graph that you want to map node.id from
# #' @param gap (default = 50) integer for largest gap size to close
# #' @param verbose (default = TRUE) logical for printing progress messages
# #' @return gWalk object
# #' @author andrew ma
# reads2node = function(alignments = NULL, gg = NULL, gap = 50, verbose = TRUE)
# {
#   if(!inherits(alignments, 'GRangesList') && !inherits(alignments, 'GRanges')) stop("alignments must be a GRangesList or GRanges object")
#   if(!inherits(gg, 'gGraph')) stop("gg must be a gGraph object")

#   if(verbose) message("converting reads > walks...")
#   raw.gws <- alignments2gw(alignments)

#   if(verbose) message(sprintf("...filling in %s bp gaps in the read walk", gap))
#   pad.gws <- pad.walks(raw.gws, gap.thresh = gap)

#   if(verbose) message("...annotating with ref gg node.ids")
#   ann.gr <- gr.val(grl.unlist(pad.gws$grl), gg$nodes$gr, val = "node.id", FUN = unique)
#   mcols(ann.gr)$map.node.id = mcols(ann.gr)$node.id                                         # not necessary but if you want to label with diff node id's
#   mcols(ann.gr)$map.snode.id = sign(mcols(ann.gr)$snode.id) * mcols(ann.gr)$map.node.id     #
#   ann.gws <- gW(grl = split(ann.gr, ann.gr$qname))
  
#   if(verbose) message("...simplifying")
#   simp.gws <- ann.gws$copy$simplify(by = "map.node.id")   # node ids get mismapped during reduction

#   return(simp.gws)
# }



# #' @name drop.nodes.walk
# #' @title drop.nodes.walk
# #'
# #' @description
# #' takes in a gWalk object and drops nodes that are below a specified width threshold
# #' 
# #' @param gws gWalk object
# #' @param width.thresh (default = 50) integer for minimum node width to keep
# #' @return gWalk object
# #' @author andrew ma
# drop.nodes.walk <- function(gws = NULL, width.thresh = 50)
# {
#   if(!inherits(gws, 'gWalk')) stop("gws must be a gWalk object")

#   gws.gr <- grl.unlist(gws$grl)
#   mcols(gws.gr)$width <- width(gws.gr)
  
#   new.gws.gr <- gws.gr[mcols(gws.gr)$width > width.thresh]
#   mcols(new.gws.gr)$width <- NULL
  
#   new.gws <- gW(grl = split(new.gws.gr, new.gws.gr$qname))
  
#   return(new.gws)
# }



# #' @name alignments2gg.d
# #' @title alignments2gg.d
# #'
# #' @description
# #' alignments2gg but can drop nodes below a certain width threshold at gWalk level
# #' 
# #' @param tile GRanges of tiles
# #' @param juncs Junction object or grl coercible to Junctions object
# #' @param genome seqinfo or seqlengths
# #' @return list with gr and edges which can be input into standard gGnome constructor
# #' @author Marcin Imielinski, Joe DeRose, Xiaotong Yao, andrew ma
# alignments2gg.d = function(alignment, width.thresh = 50, verbose = TRUE)
# {

#   if (inherits(alignment, 'GRangesList') | inherits(alignment, 'CompressedGRangesList')){
#       alignment = grl.unlist(alignment)
#   }
#   if (!inherits(alignment, 'GRanges') || !all(c('qname', 'cigar', 'flag') %in%  names(values(alignment))))
#     stop('alignment input must be GRanges with fields $qname $cigar and $flag')

#   if (verbose)
#     message('making cgChain')

#   cg = gChain::cgChain(alignment)

#   if (verbose)
#     message('disjoining query ranges and lifting nodes to reference')

#   lgr = gChain::links(cg)$x
#   verboten = c("seqnames", "ranges",
#     "strand", "seqlevels", "seqlengths", "isCircular", "start", "end",
#     "width", "element")
#   values(lgr) = cbind(values(lgr), values(cg)[, setdiff(names(values(cg)), verboten)])
#   grc = gr.disjoin(grbind(lgr, si2gr(gChain::links(cg)$x)))
#   grc$qname = seqnames(grc)
#   gwc = gW(grl = split(grc, seqnames(grc)))

#   ## lol let's just try this...
#   gwc <- drop.nodes.walk(gwc, width.thresh = width.thresh)

#   nodes = gwc$graph$nodes
#   grr = gChain::lift(cg, nodes$gr)
#   grr$insertion = FALSE

#   ## add a pad either to the right or left (basically, there should always be mapped sequence on one side on an insertion ..
#   ## otherwise there is no alignment (ie pure insertions means no alignment)
#   ix = setdiff(nodes$gr$node.id, grr$node.id)
#   if (length(ix))
#   {
#     insertions = nodes[ix]$gr
#     start(insertions) = ifelse(start(insertions)>1, start(insertions)-1, start(insertions))
#     end(insertions) = ifelse(start(insertions)== 1 & end(insertions) < seqlengths(insertions)[as.character(seqnames(insertions))],
#                              end(insertions)+1, end(insertions))
#     insertions$insertion = TRUE
#     grr = grbind(grr, gChain::lift(cg, insertions)) ## add the lifted insertions to the pile of intervals
#   }

#   ugrr = unique(gr.stripstrand(grr))

#   ## there may be dups here if say the lift aligns the contig to both the negative and positive side
#   ## of a contig
#   grr$ugrr.id = match(gr.stripstrand(grr), ugrr)
#   grr$grr.id = 1:length(grr)

#   if (any(ugrr$insertion))
#   {
#     width(ugrr[ugrr$insertion]) = 0
#   }

#   ## find insertions ie nodes that did not survive the lift
#   edges = gwc$graph$edges$dt

#   if (verbose)
#     message('lifting edges to reference')


#   ## to lift to genome cordinates, merge old edges with new ids (will duplicate edges across multimaps)
#   edges.new = edges %>% merge(gr2dt(grr), by.x = 'n1', by.y = 'node.id', allow.cartesian = TRUE) %>% merge( gr2dt(grr), by.x = 'n2', by.y = 'node.id', allow.cartesian = TRUE)
#   edges.new[, n1 := ugrr.id.x] ## we map to the 
#   edges.new[, n2 := ugrr.id.y]

#   ## flip sides for nodes that are flipped (i.e. negative strand) during lift
#   .flip = function(x) c(left = 'right', right = 'left')[x]

#   edges.new$n1.side = ifelse(strand(grr)[edges.new$grr.id.x] == '-', .flip(edges.new$n1.side), edges.new$n1.side)
#   edges.new$n2.side = ifelse(strand(grr)[edges.new$grr.id.y] == '-', .flip(edges.new$n2.side), edges.new$n2.side)

#   ## now just need to replace any edges to and from an insertion
#   ugrr$loose.left = ugrr$loose.right = NULL

#   if (verbose)
#     message('building graph')
  
#   return(list(nodes = ugrr, edges = edges.new[, .(n1, n1.side, n2, n2.side)]))
# }



#' @name get_readL
#' @title get_readL
#'
#' @description
#' outputs read length distribution and mean read length from reads (just uses $qwidth param)
#' 
#' @param reads GRanges or GRangesList of reads
#' @return list with read length distribution and mean read length
#' @author andrew ma
#' @export
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
  reads.gr <- reads.gr[!is.na(mcols(reads.gr)$qwidth)]      # first remove NA qwidth entries
  u.reads.gr <- reads.gr[!duplicated(mcols(reads.gr)$qname)]    # dedup by qname
  read.lengths <- mcols(u.reads.gr)$qwidth
  mean.read.length <- mean(read.lengths)

  read.lengths.ls <- read.lengths
  names(read.lengths.ls) <- mcols(u.reads.gr)$qname

  return(list(read.lengths.ls = read.lengths.ls, mean.read.length = mean.read.length))
}



# #' @name snap2bps
# #' @title snap2bps
# #' 
# #' @description
# #' snaps a GRanges/GRangesList to a set of breakpoints (bps) within a supplied threshold (pad)
# #' 
# #' @param gr GRanges or GRangesList to be snapped
# #' @param bps GRanges of breakpoints to snap to
# #' @param pad (default = 5) integer for snapping threshold
# #' @return GRanges or GRangesList of snapped ranges
# #' @author andrew ma
# snap2bps <- function(gr, bps, pad = 5)
# {
#   if(inherits(gr,'GRangesList') | inherits(gr, 'CompressedGRangesList')){
#     gr <- grl.unlist(gr)
#   }
#   else if(!inherits(gr, 'GRanges')){
#     stop("gr must be a GRanges or GRangesList object")
#   }

#   if(!inherits(bps, 'GRanges')){
#     stop("bps must be a GRanges object")
#   }

#   # make sure breakpoints are strandless and unique, then pad
#   bps <- bps %>% gr.stripstrand() %>% unique()
#   bpgr <- bps + pad
  
#   lgr <- gr.start(gr, ignore.strand = TRUE)     # rightmost base of segments
#   rgr <- gr.end(gr, ignore.strand = TRUE)       # leftmost base of segments

#   lov <- gr.findoverlaps(lgr, bpgr, ignore.strand = TRUE)
#   mcols(lov)$type <- "left"
#   rov <- gr.findoverlaps(rgr, bpgr, ignore.strand = TRUE)
#   mcols(rov)$type <- "right"

#   # combine overlaps and snap to breakpoints
#   ov <- grbind(lov, rov) %>% gr2dt()
#   ov[, bp := start(bps)[subject.id]]
#   ov[, bpdist := start - bp]
#   # if multiple breakpoints are within the pad width, snap to the closest one
#   ov[, multi:=.N>1, by = query.id]
#   ov[multi==T, bpdist := fifelse(abs(bpdist) == min(abs(bpdist)), bpdist, 0), by = query.id]
#   snap <- ov[bpdist != 0]
  
#   snap <- snap[, .SD[which.min(abs(bpdist))], by = query.id]
#   gdt <- gr2dt(gr)
#   gdt[, idx := .I]
  
#   gdt[snap[type == "left"], on = .(idx == query.id), start := i.bp]
#   gdt[snap[type == "right"], on = .(idx == query.id), end := i.bp + 1]

#   new.gr <- dt2gr(gdt)

#   return(new.gr)
# }



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
#' @export
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
#' @export
get.freeze <- function(gg)
{
  if(!inherits(gg, 'gGraph')) stop("gg must be a gGraph object")

  samp.nodes <- as.vector(as.matrix(gg$edges$dt[gg$edges$dt$type=="ALT",.(n1,n2)]))
  freeze.nodes <- setdiff(gg$nodes$dt$node.id,samp.nodes)

  return(freeze.nodes)
}



#' @name map.fine
#' @title map.fine
#' 
#' @description
#' maps gWalks to a finer reference gGraph's node.id's (e.g. read-disjoined ref graph)
#' and returns a new gWalk object with the finer node.id's
#' 
#' @param gws gWalk object to be mapped
#' @param gg gGraph object to map to
#' @param return.gw logical, if TRUE returns a new gWalk object, if FALSE returns a list of mapped snode.id's
#' @param minsize minimum size of a node to be included in the output
#' @return list of mapped snode.id's or a new gWalk object
#' @author andrew ma
#' @export
map.fine <- function(gws, gg, return.gw = FALSE)
{
  if(!inherits(gws, 'gWalk')) stop("gws must be a gWalk object")
  if(!inherits(gg, 'gGraph')) stop("gg must be a gGraph object")

  qq <- grl.unlist(gws$grl)     # coarse nodes, grl.ix = read walk, grl.iix = node in walk

  ov <- gr.findoverlaps(qq, gg$nodes$gr, ignore.strand = TRUE)    # we ignore gg's strands

  str <- as.character(strand(qq))[ov$query.id]
  strand(ov) <- str
  ov$walk <- qq$grl.ix[ov$query.id]   # walk id number
  ov$qname <- qq$qname[ov$query.id]   # walk name
  ov$pos <- qq$grl.iix[ov$query.id]   # position in walk
  ov$node.id <- gg$nodes$gr$node.id[ov$subject.id]   # node id in fine graph

  # now order by walk, coarse node in walk, then orient finer node id's by strand of coarse node
  k <- fifelse(str == "-", -start(ov), start(ov))
  ov <- ov[order(ov$walk, ov$pos, k)]

  # add sign back in for snode.id's
  ov$snode.id <- fifelse( as.character(strand(ov))=="-",-ov$node.id,ov$node.id )

  # return snode.id's
  ovdt <- gr2dt(ov)
  snode.idlist = split(ovdt$snode.id, ovdt$qname)

  out <- snode.idlist

  if(return.gw){
    # instantiating a new gWalk object will overwrite the snode.ids
    # instead, we will store the mapped ids in $map.snode.id
    ov$map.snode.id <- ov$snode.id
    out <- gW(grl = split(ov, ov$qname))
  }

  return(out)
}



#' @name jac
#' @title jac
#' 
#' @description
#' computes the Jaccard index between two sets
#' 
#' @param a (vector) first set
#' @param b (vector) second set
#' @return Jaccard index
#' @export
jac <- function(a, b) length(intersect(a, b)) / length(union(a, b))



# #' @name snap2bps2
# #' @title snap2bps2
# #'
# #' @description
# #' snaps GRanges/GRangesList segment ends onto graph node boundaries implied by a set of
# #' stranded breakpoints (bps), within `pad`. lower (`start`) and upper (`end`) ends snap
# #' independently, EXCEPT when a segment is narrower than `pad` and both ends are in range --
# #' then only the closer end snaps, so a tiny node near one bp isn't collapsed/inverted by
# #' both ends chasing the same boundary.
# #'
# #' @param gr GRanges or GRangesList to be snapped
# #' @param bps GRanges of stranded breakpoints (e.g. gg$junctions$breakpoints)
# #' @param pad (default = 5) integer snapping threshold
# #' @return GRanges of snapped ranges
# #' @author andrew ma and claude
# snap2bps2 <- function(gr, bps, pad = 5)
# {
#   if (inherits(gr, 'GRangesList') | inherits(gr, 'CompressedGRangesList')) {
#     gr <- grl.unlist(gr)
#   } else if (!inherits(gr, 'GRanges')) {
#     stop("gr must be a GRanges or GRangesList object")
#   }
#   if (!inherits(bps, 'GRanges')) stop("bps must be a GRanges object")

#   # --- breakpoints -> node-boundary coordinates (STRAND-AWARE) --------------
#   #   '+' bp at p  => node ENDS at p,   node STARTS at p+1
#   #   '-' bp at p  => node STARTS at p, node ENDS   at p-1
#   # >>> if slivers persist specifically on '-' junctions, this ±1 is flipped -- swap it.
#   bp.dt <- data.table(seqnames = as.character(seqnames(bps)),
#                        pos      = start(bps),
#                        str      = as.character(strand(bps)))
#   node.ends   <- rbind(bp.dt[str == "+", .(seqnames, b = pos)],
#                        bp.dt[str == "-", .(seqnames, b = pos - 1L)])
#   node.starts <- rbind(bp.dt[str == "-", .(seqnames, b = pos)],
#                        bp.dt[str == "+", .(seqnames, b = pos + 1L)])
#   node.ends[,   b.match := b]; setkey(node.ends,   seqnames, b)
#   node.starts[, b.match := b]; setkey(node.starts, seqnames, b)

#   # --- nearest legal boundary for each end ----------------------------------
#   gdt <- gr2dt(gr)
#   gdt[, `:=`(sn = as.character(seqnames), start0 = start, end0 = end)]

#   if (nrow(node.starts))
#     gdt[, ssnap := node.starts[.(sn, start), on = .(seqnames, b), roll = "nearest", x.b.match]]
#   else gdt[, ssnap := NA_integer_]
#   if (nrow(node.ends))
#     gdt[, esnap := node.ends[.(sn, end), on = .(seqnames, b), roll = "nearest", x.b.match]]
#   else gdt[, esnap := NA_integer_]

#   # distance of each end to its candidate boundary, and eligibility within pad
#   gdt[, `:=`(s.dist = abs(start - ssnap), e.dist = abs(end - esnap))]
#   gdt[, `:=`(s.elig = !is.na(ssnap) & s.dist <= pad,
#              e.elig = !is.na(esnap) & e.dist <= pad)]

#   # small-interval guard: segment narrower than pad with BOTH ends in range ->
#   # snap only the closer end (by distance-to-target); ties -> start.
#   gdt[(end - start + 1L) < pad & s.elig & e.elig & s.dist <= e.dist, e.elig := FALSE]
#   gdt[(end - start + 1L) < pad & s.elig & e.elig & s.dist >  e.dist, s.elig := FALSE]

#   gdt[(s.elig), start := ssnap]
#   gdt[(e.elig), end   := esnap]

#   # backstop: never let a snap invert or collapse a segment
#   gdt[start > end, `:=`(start = start0, end = end0)]

#   drop <- intersect(c("sn","start0","end0","ssnap","esnap","s.dist","e.dist","s.elig","e.elig","width"),
#                     names(gdt))
#   gdt[, (drop) := NULL]
#   dt2gr(gdt, seqinfo = seqinfo(gr))
# }



#' @name gap_collapse
#' @title gap_collapse
#' @description
#' gap-collapse a set of width-1 breakpoints into representative cut positions
#' 
#' @param bp GRanges of width-1 breakpoints
#' @param gap (default = 50) integer for largest gap size to collapse
#' @param min.support (default = 3) integer for minimum number of breakpoints to support a cluster
#' @param ignore.strand (default = TRUE) logical for whether to ignore strand when collapsing
#' @return GRanges of collapsed breakpoints with metadata col `support` indicating number of breakpoints supporting each collapsed breakpoint
#' @export
gap_collapse <- function(bp, gap = 50, min.support = 3L, ignore.strand = TRUE) {
    stopifnot(inherits(bp, "GRanges"))
    b   <- granges(bp)                                  # drop mcols, keep seqnames/strand
    cl  <- GenomicRanges::reduce(b, min.gapwidth = gap, with.revmap = TRUE,
                                 ignore.strand = ignore.strand)
    rv  <- cl$revmap                                    # IntegerList: which breakends -> each cluster
    pos <- start(b)                                     # width-1 => start == breakpoint
    med <- vapply(rv, function(ix) as.integer(round(median(pos[ix]))), integer(1L))
    keep <- lengths(rv) >= min.support
    out  <- GRanges(seqnames(cl), IRanges(med, width = 1L),
                    strand = if (ignore.strand) "*" else strand(cl))[keep]
    out$support <- lengths(rv)[keep]
    out
}



#' @name jct_filt
#' @title jct_filt
#' 
#' @description 
#' snap read breakpoints onto authoritative anchors first, then gap-cluster the rest
#' 
#' @param bp GRanges of width-1 breakpoints
#' @param anchors GRanges of width-1 breakpoints to snap onto first
#' @param gap (default = 50) integer for largest gap size to collapse
#' @param min.support (default = 3) integer for minimum number of breakpoints to support a cluster
#' @param ignore.strand (default = TRUE) logical for whether to ignore strand when collapsing
#' @param keep.unsupported.anchors (default = TRUE) logical for whether to keep anchors with no snapped breakpoints
#' @return GRanges of collapsed breakpoints with support mcol
#' @export
jct_filt <- function(bp, anchors, gap = 50, min.support = 3L,
                                  ignore.strand = TRUE,
                                  keep.unsupported.anchors = TRUE) {
    b <- granges(bp)
    a <- granges(anchors)
    seqlengths(b) <- NA       # tends to error, seqlengths unnecessary anyways
    seqlengths(a) <- NA

    if (length(a) == 0L)                                   # no anchors -> plain collapse
        return(gap_collapse(b, gap, min.support, ignore.strand))

    ## nearest anchor for each read breakpoint, then absorb those within `gap`
    nn  <- nearest(b, a, ignore.strand = ignore.strand)    # index into `a`, NA if none
    d   <- rep(Inf, length(b))
    hit <- !is.na(nn)
    d[hit] <- distance(b[hit], a[nn[hit]], ignore.strand = ignore.strand)
    absorbed <- hit & d <= gap

    ## anchors keep their EXACT position; support = # reads that snapped to them
    a$support <- tabulate(nn[absorbed], nbins = length(a))
    a$anchor  <- TRUE
    anchor.out <- if (keep.unsupported.anchors) a else a[a$support > 0L]

    ## leftover reads (not near any anchor) cluster among themselves as before
    novel <- gap_collapse(b[!absorbed], gap = gap, min.support = min.support,
                          ignore.strand = ignore.strand)
    if (length(novel) > 0) {
        novel$anchor <- FALSE
    }

    c(anchor.out, novel)
}


