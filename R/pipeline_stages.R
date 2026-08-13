#' @import data.table
#' @importFrom parallel mclapply
#' @importFrom stats median
#' @importFrom bamUtils read.bam
#' @import gUtils
#' @import gGnome
#' @import fastKar

#' @export
event_graph <- function(pr, eid, pairs.dt, ftpad = 1e4) {
    message("importing graph for ", pr, " event ", eid,"...")
    pr.gg <- readRDS(pairs.dt[pair == pr, complex])
    e.gr <- pr.gg$meta$events[ev.id == eid]$footprint %>%
            parse.gr(seqlengths = hg_seqlengths(chr = FALSE)) + ftpad
    message("disjoining event...")
    e.gg <- pr.gg$copy$disjoin(e.gr) %&% e.gr
    e.gg$simplify()
    loosefix(e.gg)
    list(e.gg = e.gg, e.gr = e.gr)
}

#' @export
read_walk <- function(pr, e.gr, pairs.dt, pad = 50) {
    message("importing reads...")
    e.r  <- read.bam(pairs.dt[pair == pr, tumor_bam_ont], gr = e.gr,
                     stripstrand = FALSE, isPaired = FALSE, pairs.grl = FALSE,
                     ignore.indels = FALSE, tag = c("SA"))
    e.rs <- smooth.cigar(e.r, smooth.thresh = pad)
    message("converting to walks...")
    e.gw <- alignments2gw(e.rs, ignore.overlaps = TRUE)$simplify(by = "qname")
    list(e.rs = e.rs, e.gw = e.gw)
}

#' @export
disjoin_graph <- function(e.gg, e.gw, pad = 50) {
    e.r.gg  <- e.gw$graph
    message("filtering junctions...")
    dis.bps <- jct_filt(bp = gr.start(e.r.gg$junctions[type == "ALT"]$breakpoints),
                        anchors = gr.start(e.gg$junctions[type == "ALT"]$breakpoints),
                        gap = pad, min.support = 3, ignore.strand = TRUE,
                        keep.unsupported.anchors = TRUE)
    new.bps.n <- sum(mcols(dis.bps)$anchor==F)
    message(paste0("introducing ",new.bps.n," new breakpoints..."))
    dis.nodes <- gr.breaks(bps = dis.bps, query = e.gg$nodes$gr) %>%
                 gr.stripstrand %>% .[, c()]
    e.dis <- edgefix(gg = e.gg$copy$disjoin(dis.nodes), ref.gg = e.gg)
    list(e.dis = e.dis, dis.bps = dis.bps)
}

#' @export
sample_kars <- function(e.dis, n = 1e3, seed = 1, freeze = TRUE) {
    fn <- NULL
    if(freeze){fn <- get.freeze(e.dis)}
    set.seed(seed)
    e.kars <- sample.gwalks(e.dis, N = n, frozen.nodes = fn,
                            remove.dups = TRUE, onlyhash = FALSE, keep.circular = FALSE)
    names(e.kars) <- paste0("kar_", seq_along(e.kars))
    message(paste0("sampled ", length(e.kars), " karyotypes"))
    e.kars
}

#' @export
map_reads <- function(e.gw, e.dis) {
    message("mapping read nodes to disjoined graph...")
    e.map.gw     <- map.fine(e.gw, e.dis, return.gw = TRUE)
    message("getting words...")
    e.map.snodes <- lapply(e.map.gw$grl, function(gr) gr$map.snode.id)
    names(e.map.snodes) <- e.map.gw$dt$name
    e.map.words  <- lapply(e.map.snodes, function(s) paste0(s, collapse = "|"))
    list(gw = e.map.gw, snodes = e.map.snodes, words = e.map.words)
}

#' @export
probdists <- function(e.kars, readLs, obs_words, minsize = 0, mc.cores = 4) {
    obs <- unlist(obs_words)
    message("computing prob dists for ", length(e.kars), " karyotypes...")
    e.plist <- readdist_probdist(e.kars, readL_vec = readLs, minsize = minsize,
                                 mc.cores = mc.cores, obs_words = obs)
    names(e.plist) <- names(e.kars)
    message("checking for unmapped observed reads...")
    um.pl <- readdist_probdist(e.kars, readL_vec = readLs, minsize = minsize,
                               mc.cores = mc.cores, obs_words = NULL)
    um.n <- vapply(um.pl, function(p) sum(!obs %in% names(p)), integer(1))
    if (any(um.n > 0)) {
        message(sum(um.n > 0), " karyotypes assigned bkgd probs to observed reads ",
                "(median = ", median(um.n), " reads)")
    }
    list(plist = e.plist, unmapped = um.n, observed = length(obs))
}

#' @export
score <- function(pll, e.map.words, pr, eid) {
    obs <- unlist(e.map.words)
    data.table(pair = pr, ev.id = eid, kar = names(pll$plist),
               sumloglik = vapply(pll$plist, function(p) sum(log(p[obs])), numeric(1)),
               unmapped = pll$unmapped, observed = rep(pll$observed,length(pll$plist)))
}
