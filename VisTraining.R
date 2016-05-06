library("reshape")
library("ggplot2")

ProcessDataFile <- function(fn) {

	k = read.csv(fn, head=T)
	k_m = melt(k, id=c("iter"))

	col_to_facetid = data.frame(variable=c("diff0","diff1","diff2","diff3","diff4","diff5",
	"diff6","diff7","diff8","diff9","diff10","ypos0","ypos1","ypos2","ypos3","ypos4","ypos5",
	"ypos6","ypos7","ypos8","ypos9","tprob00","tprob01","tprob02",
	"tprob03","tprob04","tprob10","tprob11","tprob12","tprob13","tprob14",
	"tprob20","tprob21","tprob22","tprob23","tprob24","tprob30","tprob31",
	"tprob32","tprob33","tprob34","tprob40","tprob41","tprob42","tprob43",
	"tprob44"), facetid = c(rep("Error Function",11), rep("Y Coordinate of Contact Points", 10), rep("Transition Probability", 25)))

	k_m = merge(k_m, col_to_facetid, by=c("variable"))

#	g <- ggplot(k_m) + theme_bw() + geom_line(aes(x=iter, y=value, colour=variable)) + facet_wrap(~facetid, ncol=1, scales="free_y")

	list(data=k_m)
}

x = ProcessDataFile("last_training.log")
g12 <- ggplot(x$data) + theme_bw() + geom_line(aes(x=iter, y=value, color=variable)) + facet_wrap(~facetid, ncol=1, scales="free_y")
g12 <- g12 + scale_color_discrete(name="")
show(g12)
