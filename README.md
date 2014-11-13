fastRPCA
========

Matlab code for all variants of robust PCA and SPCP. This implements the code from the conference paper "A variational approach to stable principal component pursuit" by Aravkin, Becker, Cevher, Olsen; UAI 2014.

Not only is this code fast, but it is the only code we know of that solves all common stable principal component pursuit (SPCP) variants, including the new variants we introduced in the paper. All these variants are in some sense equivalent, but some of them are easier to solve, and some have parameters that are easier to estimate.  See the [paper for details](http://arxiv.org/abs/1406.1089)


<p align="center"><img src="http://amath.colorado.edu/faculty/becker/escalatorImage.jpg" /></p>


More info on robust PCA and stable principal component pursuit
(websites with software, review articles, etc.)

* ["A variational approach to stable principal component pursuit"](http://arxiv.org/abs/1406.1089)
* [LRS Library](https://github.com/andrewssobral/lrslibrary)
* [Matrix Factorization jungle](https://sites.google.com/site/igorcarron2/matrixfactorizations)
* [One of the first papers on RPCA](http://arxiv.org/abs/0912.3599)


Citation
---------
```
@inproceedings{aravkin2014,
    author       = "Aravkin, A. and Becker, S. and Cevher, V. and Olsen, P.",
    title        = "A variational approach to stable principal component pursuit",
    booktitle    = "Conference on Uncertainty in Artificial Intelligence (UAI)",
    year         = "2014",
    month        = "July",
}
```

Code and installation
----

The code runs on MATLAB and does not require any mex files or installation. Just unzip the file and you are set. Run `setup_fastRPCA` to set the correct paths, and try the `demos` directory for sample usage of the code.

Authors
--------------
The code was developed by all the authors of the UAI paper, but primary development is due to Stephen Becker and Aleksandr (Sasha) Aravkin. Further contributions are welcome.

## Contact us
* [Sasha Aravkin](https://sites.google.com/site/saravkin), IBM Research
* [Stephen Becker](amath.colorado.edu/faculty/becker/), University of Colorado Boulder

