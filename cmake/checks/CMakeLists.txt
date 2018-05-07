FILE(GLOB_RECURSE files "*cmake")
foreach(var ${files})
  include("${var}")
endforeach(var)
