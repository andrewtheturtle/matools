#' @import data.table
#' @importFrom parallel mclapply
#' @importFrom stats median
#' @importFrom bamUtils read.bam
#' @import gUtils
#' @import gGnome
#' @import fastKar
#' @import gTrack

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
    e.dis <- edgefix(gg = e.gg$copy$disjoin(dis.nodes))
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

#' @export
load_event <- function(pr, eid, prdt){
    if (is.character(prdt)) prdt <- readRDS(prdt)          # accept path or table
    r <- prdt[pair == pr & ev.id == eid]
    stopifnot(nrow(r) == 1L)
    g   <- readRDS(r$event_g)      # list(e.gg, e.gr)
    rd  <- readRDS(r$read_gws)     # list(e.rs, e.gw)
    dj  <- readRDS(r$disjoin)      # list(e.dis, dis.bps)
    pll <- readRDS(r$probdists)    # list(plist, unmapped, observed)
    ev <- list(pr=r$pair, eid=r$ev.id,
               e.gg=g$e.gg, e.gr=g$e.gr,
               e.gw=rd$e.gw, e.rs=rd$e.rs,
               e.dis=dj$e.dis, dis.bps=dj$dis.bps,
               e.kars=readRDS(r$samp_kars),
               map=readRDS(r$map),                      # list(gw, snodes, words)
               e.plist=pll$plist, pll=pll,              # $plist
               e.ll.dt=fread(r$loglik))
    stopifnot(length(ev$e.kars) == length(ev$e.plist),
              length(ev$e.plist) == nrow(ev$e.ll.dt))   # kars x plist x ll aligned
    ev
}

#' @export
plot_event_matrix <- function(pr, eid, prdt, plotdir){
    e  <- load_event(pr, eid, prdt)
    fn <- function(tag, ext) paste0(plotdir, tag, "_", pr, "_", eid, ext)

    # disjoin
    p.gt <- c(e$e.gg$gtrack(name="e.gg", labels.suppress.grl=TRUE),
              e$e.gw$gtrack(name="e.gw", labels.suppress.grl=TRUE),
              e$e.dis$gtrack(name="e.dis", labels.suppress.grl=TRUE))
    ppdf(plot(p.gt, e$e.gr), width=10, height=14, filename=fn("disjoin",".pdf"))

    # disjoin zoom (only if new, non-anchor breakpoints exist)
    if (any(mcols(e$dis.bps)$anchor == FALSE)) {
        p.gr <- e$dis.bps[mcols(e$dis.bps)$anchor == FALSE] + 1e2
        ppdf(plot(p.gt, p.gr), width=10, height=14, filename=fn("disjoin-zoom",".pdf"))
    }

    # karyotypes
    kar.h <- max(14, 2 + 0.4 * length(e$e.kars))
    e$e.dis$nodes$mark(label = e$e.dis$nodes$dt$node.id)
    lapply(e$e.kars, function(k) k$nodes$mark(label = k$nodes$dt$node.id))
    p.w  <- lapply(seq_along(e$e.kars), function(j)
              e$e.kars[[j]]$gtrack(name=names(e$e.kars)[j], gr.colorfield="label",
                                   cex.label=2, labels.suppress.grl=TRUE))
    p.gt <- c(e$e.dis$gtrack(name="e.dis", gr.colorfield="label", cex.label=2,
                             labels.suppress.grl=TRUE), do.call(c, p.w))
    ppdf(plot(p.gt, e$e.gr), width=10, height=kar.h, filename=fn("karyotypes",".pdf"))

    # read-length dist (read.lengths.ls is a named vector — no unlist)
    rdst <- get_readL(e$e.rs)
    ravg <- round(rdst$mean.read.length)
    r.hist <- ggplot(data.frame(len=rdst$read.lengths.ls), aes(x=len)) +
              geom_histogram(bins=50) +
              geom_vline(xintercept=ravg, color="red", linetype="dashed") +
              scale_x_log10() +
              labs(title=paste0(pr,", ev ",eid," read len"), x="read length (bp)", y="count")
    ggsave(r.hist, filename=fn("readL_dist",".png"), width=5, height=4)
    invisible(NULL)
}

#' @export
rank_kars <- function(pr, eid, prdt, n = 10){
    e <- load_event(pr, eid, prdt)
    head(e$e.ll.dt[order(-sumloglik), .(kar, sumloglik, unmapped, observed)], n)
}

#' @export
compare_kars <- function(pr, eid, prdt, rank.a = 1, rank.b = -1,
                         PLOTDIR, NSAMP = 1000){
    e <- load_event(pr, eid, prdt)
    ord <- order(-e$e.ll.dt$sumloglik)                                      # rank 1 = best
    idx <- function(r) if (r < 0) ord[length(ord) + r + 1] else ord[r]      # -1 = worst
    ia <- idx(rank.a)
    ib <- idx(rank.b)
    ka <- e$e.ll.dt$kar[ia]
    kb <- e$e.ll.dt$kar[ib]                        # names retained
    fn <- function(tag, ext) paste0(PLOTDIR, tag, "_", pr, "_", eid, ext)

    p <- e$e.plist[[ka]]
    q <- e$e.plist[[kb]]
    ravg <- round(get_readL(e$e.rs)$mean.read.length)

    # per-read LLR (a vs b)
    d       <- log(p) - log(q)
    dll.dt  <- data.table(dll = d[unlist(e$map$words)], word = unlist(e$map$words))
    ggsave(ggplot(dll.dt, aes(x=dll)) + geom_histogram(bins=30) +
             labs(title=paste0("LLR ", ka, "/", kb), x="llr", y="count"),
           filename=fn("loglik_hist",".png"), width=7, height=4)

    # show top differentiating reads vs. uninformative reads
    # canonical key invariant to RC (RC = -rev): min(fwd, -rev) as pipe-string
    rc_canon <- function(s) {
        f <- paste(s,       collapse="|")
        r <- paste(-rev(s), collapse="|")
        if (f <= r) f else r
    }
    # rank words by dll, collapse RC pairs to one canonical orientation
    word.dll <- d[unique(unlist(e$map$words))]          # named: word -> dll (unique words)
    canon    <- vapply(names(word.dll),
                       function(w) rc_canon(as.integer(strsplit(w, "\\|")[[1]])),
                       character(1))
    ord      <- order(-word.dll)
    top.words   <- names(word.dll)[ord][!duplicated(canon[ord])][1:5]
    uninf.words <- names(word.dll)[order(abs(word.dll))]
    uninf.words <- setdiff(uninf.words, top.words)[1:5]
    # first read matching each chosen word
    first_read <- function(w) names(e$map$words)[match(w, unlist(e$map$words))]
    top.reads   <- vapply(top.words,   first_read, character(1))
    uninf.reads <- vapply(uninf.words, first_read, character(1))
    top.gw   <- e$map$gw[name %in% top.reads]
    uninf.gw <- e$map$gw[name %in% uninf.reads]

    top.gw$nodes$mark(label = top.gw$nodes$dt$map.node.id)                  # highlight informative reads
    uninf.gw$nodes$mark(label = uninf.gw$nodes$dt$map.node.id)
    e$e.kars[[ka]]$nodes$mark(label = e$e.kars[[ka]]$nodes$dt$node.id)
    e$e.kars[[kb]]$nodes$mark(label = e$e.kars[[kb]]$nodes$dt$node.id)
    
    p1.gt <- c(e$e.kars[[kb]]$gtrack(name=paste0("rank ",rank.b," (",kb,")"), cex.label=2, gr.colorfield="node.id", labels.suppress.grl=TRUE),
              top.gw$gtrack(name="top reads", cex.label=2, gr.colorfield="map.node.id", labels.suppress.grl=TRUE),
              e$e.kars[[ka]]$gtrack(name=paste0("rank ",rank.a," (",ka,")"), cex.label=2, gr.colorfield="node.id", labels.suppress.grl=TRUE))
    p2.gt <- c(e$e.kars[[kb]]$gtrack(name=paste0("rank ",rank.b," (",kb,")"), cex.label=2, gr.colorfield="node.id", labels.suppress.grl=TRUE),
              uninf.gw$gtrack(name="uninformative reads", cex.label=2, labels.suppress.grl=TRUE),
              e$e.kars[[ka]]$gtrack(name=paste0("rank ",rank.a," (",ka,")"), cex.label=2, gr.colorfield="node.id", labels.suppress.grl=TRUE))
    ppdf(plot(p1.gt, e$e.gr), width=10, height=14, filename=fn("topreads",".pdf"))
    ppdf(plot(p2.gt, e$e.gr), width=10, height=14, filename=fn("uninf-reads",".pdf"))

    # sampled-read null: does observed separation exceed chance
    N.reads <- length(e$e.gw)
    llr <- function(s) sum(log(p[s])) - sum(log(q[s]))
    pr_ratios <- vapply(1:NSAMP, function(x) llr(sample(names(p), N.reads, prob=p, replace=TRUE)), numeric(1))
    qr_ratios <- vapply(1:NSAMP, function(x) llr(sample(names(q), N.reads, prob=q, replace=TRUE)), numeric(1))
    err <- (sum(pr_ratios <= 0) + sum(qr_ratios >= 0)) / (2*NSAMP)
    pq  <- data.table(ratio=c(pr_ratios,qr_ratios), type=rep(c(ka,kb), each=NSAMP))
    ggsave(ggplot(pq, aes(x=ratio, fill=type)) +
             geom_histogram(bins=30, position="identity", alpha=0.5) +
             geom_vline(xintercept=sum(dll.dt$dll), color="black", linetype="dashed") +
             annotate("label", x=-Inf, y=Inf, label=paste0("samples=",NSAMP,"\nreads=",N.reads,"\nerr=",round(err,4)),
                      hjust=0, vjust=1, size=8/.pt) +
             labs(title="sampled-read LLR null", x=paste0("llr ",ka,"/",kb), y="count"),
           filename=fn("samp_llr_hist",".png"), width=7, height=4)

    list(pr=pr, eid=eid, kar.a=ka, kar.b=kb, rank.a=rank.a, rank.b=rank.b,
         obs.sll=sum(dll.dt$dll), error_rate=err, N.reads=N.reads)
}

#' @export
score_others <- function(pr, eid, prdt, pairs.dt, rank.self = 1,
                         PLOTDIR, N_WALKS = 100, seed = 1){
    e <- load_event(pr, eid, prdt)
    others.prs <- setdiff(prdt$pair, pr)

    message(pr,"_",eid," loaded, gathering graphs + building union partition...")
    o.raw <- setNames(lapply(others.prs, function(opr){
        og <- readRDS(pairs.dt[pair == opr, complex])
        og$simplify()
        g <- og$copy$disjoin(e$e.gr) %&% e$e.gr
        # tryCatch({loosefix(g)},error=function(e){message(opr,"\n",e)})            # loosefix breaking on cn.x not found
        g
    }), others.prs)
    raw.graphs <- c(list(e$e.dis), o.raw)
    all.bps <- do.call(grbind, lapply(raw.graphs, function(g){
                   g$junctions$breakpoints})) %>% gr.stripstrand %>% unique
    union.nodes <- gr.breaks(bps = all.bps, query = e$e.dis$nodes$gr) %>%
                   gr.stripstrand %>% .[, c()]

    disjoin_on_union <- function(gg) {
        g <- gg$copy$disjoin(union.nodes)
        if(length(g$edges$dt$cn)==0 || is.null(g$edges$dt$cn)){
            origin.nodes <- g$edges$dt$n1
            g$edges$mark(cn = g$nodes[origin.nodes]$dt$cn)
        }
        g <- edgefix(g)
    }
    self.dis  <- disjoin_on_union(e$e.dis)
    other.dis <- lapply(o.raw, function(g) disjoin_on_union(g))

    message("plotting union disjoin graphs...")
    p.gt <- c(self.dis$gtrack(name=pr, labels.suppress.grl=TRUE),
              do.call(c,lapply(seq_along(other.dis), function(i) other.dis[[i]]$gtrack(name=names(other.dis)[i], labels.suppress.grl=TRUE))))
    ppdf(plot(p.gt, e$e.gr), width=10, height=14, filename=paste0(PLOTDIR, "master_disjoin_", pr, "_", eid, ".pdf"))

    message("re-mapping reads onto union partition...")
    map.u <- map_reads(e$e.gw, self.dis)
    obs   <- unlist(map.u$words)

    message("re-sampling + ranking self karyotypes with new words...")
    self.gw <- sample_kars(self.dis, n = N_WALKS, seed = seed, freeze = TRUE)       # freeze = TRUE to avoid shuffling ref-only nodes
    if (length(self.gw) == 1L) {
        k.self <- self.gw[[1]]
    } else {
        self.pd  <- readdist_probdist(self.gw,
                        readL_vec = get_readL(e$e.rs)$read.lengths.ls,
                        minsize = 0, mc.cores = 4, obs_words = obs)
        self.sll <- vapply(self.pd, function(kp) sum(log(kp[obs])), numeric(1))
        k.self   <- self.gw[[ order(-self.sll)[rank.self] ]]                        # rank 1 = best self kar
    }

    message("now sampling + scoring other events...")
    doc <- list()
    others <- lapply(others.prs, function(opr){
        o.dis <- other.dis[[opr]]
        fn <- get.freeze(o.dis)                                                     # node.ids with only ref edges
        if (length(fn) == length(o.dis$nodes$gr)) {                                 # all ref-only -> nothing to shuffle
            message("all nodes frozen for ", opr, ", generating 1 karyotype...")
            nW <- 1L; FREEZE <- FALSE
        } else {
            message("ALT jcts present for ", opr, ", sampling ", N_WALKS, " karyotypes...")
            nW <- N_WALKS; FREEZE <- TRUE
        }
        gw <- sample_kars(o.dis, n = nW, seed = seed, freeze = FREEZE)
        if (length(gw) != 1L) {
            message("multiple karyotypes for ", opr, ", picking one at random...")
            set.seed(seed); pick <- sample(seq_along(gw), 1)
            doc[[opr]] <<- data.table(pair=pr, ev.id=eid, others=opr,
                                      n.kars=length(gw), picked=pick)
            gw <- gw[[pick]]
        } else gw <- gw[[1]]
        gw
    })
    names(others) <- others.prs

    master <- c(setNames(list(k.self), pr), others)
    lapply(master, function(k) k$nodes$mark(label = k$nodes$dt$node.id))
    message("scoring master set...")
    pd <- readdist_probdist(master,
                            readL_vec = get_readL(e$e.rs)$read.lengths.ls,
                            minsize = 0, mc.cores = 4, obs_words = obs)
    names(pd) <- names(master)
    sll <- vapply(pd, function(kp) sum(log(kp[obs])), numeric(1))

    message("plotting...")
    p.lab <- do.call(c, lapply(seq_along(master), function(i){
        master[[i]]$gtrack(name=names(master)[i], cex.label=2,
                           labels.suppress.grl=TRUE)}))
    ppdf(plot(p.lab, e$e.gr), width=10, height=14,
         filename=paste0(PLOTDIR, "master_kars_", pr, "_", eid, ".pdf"))
    p.nol <- do.call(c, lapply(seq_along(master), function(i){
        master[[i]]$gtrack(name=names(master)[i], cex.label=2,
                           labels.suppress.grl=TRUE, labels.suppress=TRUE)}))
    ppdf(plot(p.nol, e$e.gr), width=10, height=14,
         filename=paste0(PLOTDIR, "master_kars_unlabeled_", pr, "_", eid, ".pdf"))

    list(summary = data.table(pair=pr, ev.id=eid, sample=names(sll),
                              is.self = names(sll)==pr, rank.self=rank.self,
                              sumloglik=sll),
         multi.kar.doc = rbindlist(doc))
}
