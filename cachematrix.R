makeCacheMatrix <- function(x = matrix()) {
	inv <- NULL
	set<- function (y) {
		x <<- y
		inv <<- NULL
	}
	get <- function () x
	setinverse <- function(inverese)
	inv <<- inverse
	getinverse <- function () inv
	list(set=set, get=get, setinverse=serinverse, getinverse = getinverse)
}
## This function returns the inverse of the matrix. 
## It first checks if if the inverse has allready been computed and if so returns the allready computed value.

CacheSolve <- function (x, ...) {
	inv <- x$getinverse(
	if(!is.null(inv)) {
		return(inv)
	}
	data <- x$get(
	inv <- solve(data)
	x$setinverse(inv)
	inv
}
## This function creates a function of the provided data where if the data in not a missing value an inverse for each data point is recieved































































