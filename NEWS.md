# conleyreg 0.1.8
* Upgraded to R's current default C++17
* Enforced panels to have numeric cross-sectional and temporal identifiers

# conleyreg 0.1.7
* Adjusted to a change in how fixest objects list obsRemoved

# conleyreg 0.1.6
* `rowwise` option added
* OpenMP optional
* `par_dim` can be specified based on language

# conleyreg 0.1.5
* Major package extension
* Package allows distances matrices to be pre-computed
* C++ code parallelized and further optimized
* `dist_comp` values changed to `"spherical"` and `"planar"`
* Distance matrices primarily rely on internal, optimized C++ functions, with `sf::st_distance` only used if explicitly requested by the user
* More distance functions available
* User can choose dimension along which functions parallelize
* Poisson model added
* Non-sf data can use any projection
* `conleyreg` returns variance-covariance matrix upon request
* `conleyreg` returns goodness of fit measures upon request
* Various additional RAM optimization options available
* Function drawing random location sample added

# conleyreg 0.1.4
* Corrected a bug in serial correction of balanced panel data that led data.table to throw an error

# conleyreg 0.1.2
* Removed raster package dependency to ensure compatibility with upcoming sf package update.
