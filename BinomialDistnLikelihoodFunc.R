# R script to generate Figure 3 in "Bayesian inference: More than Bayes's theorem",
# by Loredo & Wolpert 2024.  Written by Robert L Wolpert.
#
# Arguments:
#  s = vector of possible successes, from Binomial Bi(n, th) sampling dist'n
#  n = number of trials
#  a = alpha parameter of Be(a,b) prior distribution for th = theta
#  b = beta  parameter of Be(a,b) prior distribution for th = theta
#  th= display angle of X and Y axes, for 3d appearance
#  k = granularity for plotting success probability p parameter
#  dz= vertical scale
#  inred = Which success count should be stressed, by making it red?
#  inblu = Which success probaility should be stressed, by making it blue?
#  ps= Want a postscript output?  Put the file name here.
#
#
xyz <- function(s=0:20, n=max(s), a=0.5, b=a, th=60, k=500, dz=1,
                inred=13, inblu=0.5, ps="") {
  f  <- n - s;
  th <- th * pi/180;
  sok<- s[(s + a>1) & (f + b>1)]
  fok<- n - sok;
  if(nchar(ps)) {
    postscript(ps);
  }
  opar <- par(no.readonly=TRUE);
  par(mar=(opar$mar+c(0,1,0,0)));  # Add some space on left for big ylab
  yr <- range(0,
        exp(lgamma(a + b) - lgamma(a) - lgamma(b) + lgamma(n + 1) - 
            lgamma(sok + 1) - lgamma(fok + 1) + (sok + a - 1) * log(sok + a - 1) + 
            (fok + b - 1) * log(fok + b - 1) - (n + a + b - 2) * log(n + a + b - 2)));
  xr <- c(0, 1);
  
  yl <- range(yr, yr + dz * sin(th));
  xl <- range(xr, xr + dz * cos(th));

  plot(0, xlim=xl, ylim=yl, type="n", xlab=expression( theta ),
       ylab="Probability Density", axes=F, cex.lab=2);
  segments(0, 0, dz * cos(th), dz * sin(th));
  segments(0, 0, 1, 0);
  segments(0, 0, 0, yr[2]);
  segments(0, 0, dz * cos(th), dz * sin(th));
  for(i in 1:length(s)) {
    p <- (1:(k - 1))/k;
    x <- p  +  dz * cos(th) * s[i]/n;
    y <- dz * sin(th) * s[i]/n  + 
        exp(lgamma(a + b) - lgamma(a) - lgamma(b) + lgamma(n + 1)
             - lgamma(s[i] + 1) - lgamma(f[i] + 1)
             + (s[i] + a - 1) * log(p) + (f[i] + b - 1) * log(1 - p));
    if(is.element(s[i], inred)) {
        myc<- "red";     myw=4;
    } else {
        myc <- "black";  myw=1;
    }
    lines(x, (rep(dz * sin(th) * s[i]/n, length(p))), lty=2, col=myc);
    ok <- (y>=yl[1] & y<=yl[2]);   lines(x[ok], y[ok], col=myc, lwd=myw);
    text(dz * cos(th) * (s[i] + 0.5)/n - 0.05, dz * sin(th) * (s[i] + 0.5)/n, 
         s[i], col=myc, cex=1.5);
    if(length(inblu)) {
      if(is.element(s[i], inred)) myc<- "purple" else myc <- "blue";
      myx  <- inblu + dz * cos(th) * s[i]/n;
      myy0 <- dz * sin(th) * s[i]/n;
      myy1 <- dz * sin(th) * s[i]/n  + 
              exp(lgamma(a + b) - lgamma(a) - lgamma(b) + lgamma(n + 1)
                  - lgamma(s[i] + 1) - lgamma(f[i] + 1)
                  + (s[i] + a - 1) * log(inblu) + (f[i] + b - 1) * log(1 - inblu));
      segments(myx, myy0, myx, myy1, col=myc, lwd=4);
    }
    if(length(inblu)) {segments(inblu, 0, inblu + dz * cos(th), dz * sin(th),
                                lty=2, col="blue");}
    text(c(0.0, 0.5, 1.0), -0.03, c("0.0", "0.5", "1.0"), cex=1.25);
    for(i in 0:10) { segments(i/10, 0, i/10, -0.02); }
  }
  text(0.75, sin(th) * 1.5, expression(paste( f * group("(", y, "|") * theta , ")",
  " = ", bgroup("(", atop( n, y ), ")") * ~~ theta ^ y * ~~ (1 - theta ) ^ {n-y} )) ,
  ## text(0.75, sin(th) * 1.5, expression(paste( f * group("(", x, "|") * theta , ")",
  ## " = ", bgroup("(", atop( n, x ), ")") * ~~ theta ^ x * ~~ (1 - theta ) ^ {n-x} )) ,
  cex=3);
  par(opar);
  if(nchar(ps)) {
    dev.off();
    print(paste("Look for figure in file", ps));
  }
}  

hyptest <- function(s=13, n=17, p=0.5, ab=0.5) {
  print(paste("P-value:   ", pbinom(n - s, n, 1 - p), ".", sep=""));
  print(paste("Post Prob: ", pbeta(p, ab + s, ab + n - s), ".", sep=""));
}
