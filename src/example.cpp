#include <cstdio>
#include <cstdlib>
#include "laplace.h"
#include <boost/multi_array.hpp>

int main(int argc, char** argv)
{
   if(argc<1) {
      printf("usage: %s <>\n", argv[0]);
      exit(EXIT_FAILURE);
   }

   size_t n1=4, n2=6;
   double h1=1.0, h2=1.0;
   double a1=1.0, a2=1.0;
   pde::types::boundary bdtype=pde::types::Neumann;
   double bdvalue=0.0;
   bool allow_shift = (bdtype==pde::types::Neumann ? true : false);

   // set number of threads used by the internal fft routine
   pde::fftw_threads(1);

   boost::multi_array<double,2> U(boost::extents[n1][n2]);
   boost::multi_array<double,2> F(boost::extents[n1][n2]);
   boost::multi_array<double,2> U2,F2;
   double trunc;  // return value for poisolve(), indicates truncation error

   // here we always assume that U only contains inner points and
   // boundary point values are inferred by the boundary condition,
   // in particular, we then have size of U = size of F

   // we know the solution to Laplace U = F and see if
   // poisolve() can recover it (up to a constant in case of Neumann)
   printf("\n");
   printf("define solution U, apply F=Laplace(U), then solve Poisson equation\n");
   arr::runif(U);
   pde::laplace(F,U,a1,a2,h1,h2,bdvalue,bdtype);   // F = Laplace U
   // Poisson equation solver
   trunc=pde::poisolve(U2,F,a1,a2,h1,h2,bdvalue,bdtype,false);
   // print results
   printf("U=\n");
   arr::print(U);
   printf("F=\n");
   arr::print(F);
   printf("U2=\n");
   arr::print(U2);
   printf("\n");
   // print diagnostics
   if(bdtype==pde::types::Neumann) {
      printf("expected truncation error for 0-Neumann boundary: %e\n",
             pde::neumann_error(F));
      printf("compatible boundary would have been: %e\n",
             pde::neumann_compat(F,a1,a2,h1,h2));
   }
   printf("truncation error returned by poisolve(): %f\n",trunc);
   printf("error in solution: %f\n", arr::diff(U,U2,allow_shift,false));



   // we define the rhs F, then solve the Poisson equation and
   // see if it is actually a solution to Laplace U = F
   printf("\n");
   printf("define rhs F, solve for U, see if F2=Laplace(U) equals F\n");
   arr::runif(F);
   if(bdtype==pde::types::Neumann) {
      // Poisson's equation and Neumann boundary condition in general
      // won't have a solution with a random right hand side F
      // here, find the right boundary value so that a solution exists
      bdvalue=pde::neumann_compat(F,a1,a2,h1,h2);
      printf("set compatible Neumann boundary value to %f\n",bdvalue);
   }
   // Poisson equation solver
   trunc=pde::poisolve(U,F,a1,a2,h1,h2,bdvalue,bdtype,false);
   // check how well U solves the equation
   pde::laplace(F2,U,a1,a2,h1,h2,bdvalue,bdtype);
   // print results
   printf("F=\n");
   arr::print(F);
   printf("U=\n");
   arr::print(U);
   printf("F2=\n");
   arr::print(F2);
   printf("\n");
   // print diagnostics
   printf("truncation error returned by poisolve(): %e\n",trunc);
   printf("actual truncation error:                 %e\n",
          arr::diff(F,F2,false,false));


   // minor cleanup in case fftw-threads were used 
   pde::fftw_clean();

   return EXIT_SUCCESS;
}


