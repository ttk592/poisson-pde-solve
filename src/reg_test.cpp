/*
 * consistency test of the laplace/poisson solver library
 * using the boost unit test framework
 *
 * it simply tests whether the poison solver is the inverse of
 * the laplace operator
 *
 */

#include "laplace.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE PoisonSolverTest
#include <boost/test/unit_test.hpp>



struct MyFixture {
public:
   // Laplace U = F
   boost::multi_array<double,2> U,Uinput;
   boost::multi_array<double,2> F,Finput;
   std::vector<double> bd1a, bd1b, bd2a, bd2b;
   double h1, h2;
   double a1, a2;
   bool  U_contains_boundary;
   MyFixture()
   {
      h1=0.7;
      h2=1.2;
      a1=7.3;
      a2=0.9;
      U_contains_boundary=false;
   }
   void resize(size_t n1, size_t n2)
   {
      Uinput.resize(boost::extents[n1][n2]);
      // if U contains it's own boundary then F will be of size n-2
      if(U_contains_boundary) {
         Finput.resize(boost::extents[n1-2][n2-2]);
         bd1a.resize(n2-2);
         bd1b.resize(n2-2);
         bd2a.resize(n1-2);
         bd2b.resize(n1-2);
      } else {
         Finput.resize(boost::extents[n1][n2]);
         bd1a.resize(n2);
         bd1b.resize(n2);
         bd2a.resize(n1);
         bd2b.resize(n1);
      }
   }
   void randomize()
   {
      arr::runif(Finput);
      arr::runif(Uinput);
      arr::runif(bd1a);
      arr::runif(bd1b);
      arr::runif(bd2a);
      arr::runif(bd2b);
   }

};



// test that boundary functions have been implemented correctly
BOOST_AUTO_TEST_CASE( BoundaryTest )
{

   const double eps=1e-14;
   size_t n1=140, n2=267;
   MyFixture f;

   std::vector<double> bd1a, bd1b, bd2a, bd2b;


   std::vector<bool> inner_points_only(2);
   std::vector<pde::types::boundary> boundary_type(2);
   inner_points_only[0]=false;
   inner_points_only[1]=true;
   boundary_type[0]=pde::types::Dirichlet;
   boundary_type[1]=pde::types::Neumann;

   // first set boundary of 2d-array Uinput, according to vectors
   // then retrieve boundary values back to vectors and compare
   for(size_t i=0; i<inner_points_only.size(); i++) {
      for(size_t j=0; j<boundary_type.size(); j++) {
         pde::types::boundary bdtype=boundary_type[j];
         f.U_contains_boundary=!inner_points_only[i];
         f.resize(n1,n2);
         f.randomize();
         pde::set_boundary(f.Uinput,f.h1,f.h2,f.bd1a,f.bd1b,f.bd2a,f.bd2b,
                           bdtype, !f.U_contains_boundary);
         pde::get_boundary(bd1a,bd1b,bd2a,bd2b,f.Uinput,f.h1,f.h2,bdtype);
         BOOST_CHECK_SMALL( arr::diff(bd1a,f.bd1a), eps );
         BOOST_CHECK_SMALL( arr::diff(bd1b,f.bd1b), eps );
         BOOST_CHECK_SMALL( arr::diff(bd2a,f.bd2a), eps );
         BOOST_CHECK_SMALL( arr::diff(bd2b,f.bd2b), eps );
      }
   }
   // same but with the specialised set_boundary() function where only
   // a scalar value as boundary is supplied
   for(size_t i=0; i<inner_points_only.size(); i++) {
      for(size_t j=0; j<boundary_type.size(); j++) {
         pde::types::boundary bdtype=boundary_type[j];
         f.U_contains_boundary=!inner_points_only[i];
         f.resize(n1,n2);
         f.randomize();
         double c=arr::runif();
         pde::set_boundary(f.Uinput,f.h1,f.h2,c,
                           bdtype, !f.U_contains_boundary);
         pde::get_boundary(bd1a,bd1b,bd2a,bd2b,f.Uinput,f.h1,f.h2,bdtype);
         BOOST_CHECK_SMALL( arr::diff(bd1a,c), eps );
         BOOST_CHECK_SMALL( arr::diff(bd1b,c), eps );
         BOOST_CHECK_SMALL( arr::diff(bd2a,c), eps );
         BOOST_CHECK_SMALL( arr::diff(bd2b,c), eps );
      }
   }


   // first read boundary of 2d-array Uinput, write to vectors
   // then, set boundary of 2d-array U according to vectors
   // then compare
   for(size_t j=0; j<boundary_type.size(); j++) {
      pde::types::boundary bdtype=boundary_type[j];
      f.U_contains_boundary=true;
      f.resize(n1,n2);
      f.randomize();
      f.U.resize(boost::extents[f.Uinput.shape()[0]][f.Uinput.shape()[1]]);
      f.U=f.Uinput;
      pde::get_boundary(f.bd1a,f.bd1b,f.bd2a,f.bd2b,f.Uinput,f.h1,f.h2,bdtype);
      pde::set_boundary(f.U,f.h1,f.h2,f.bd1a,f.bd1b,f.bd2a,f.bd2b,
                        bdtype, !f.U_contains_boundary);
      BOOST_CHECK_SMALL( arr::diff(f.Uinput,f.U,false,true), eps );
   }
}



// test that laplace() equals div(grad())
BOOST_AUTO_TEST_CASE( LaplaceTest )
{

   const double eps=1e-12;
   size_t n1=132, n2=221;
   MyFixture f;

   boost::multi_array<double,2> DX, DY, F2;

   std::vector<pde::types::boundary> boundary_type(2);
   boundary_type[0]=pde::types::Dirichlet;
   boundary_type[1]=pde::types::Neumann;

   // first set boundary of 2d-array Uinput, according to vectors
   // then retrieve boundary values back to vectors and compare
   for(size_t i=0; i<boundary_type.size(); i++) {
      pde::types::boundary bdtype=boundary_type[i];
      f.U_contains_boundary=false;
      f.resize(n1,n2);
      f.randomize();

      pde::laplace(f.F,f.Uinput,f.a1,f.a2,f.h1,f.h2,
                   f.bd1a,f.bd1b,f.bd2a,f.bd2b, bdtype);
      pde::grad(DX,DY,f.Uinput,sqrt(f.a1),sqrt(f.a2),f.h1,f.h2,
                f.bd1a,f.bd1b,f.bd2a,f.bd2b, bdtype);
      pde::div(F2,DX,DY,sqrt(f.a1),sqrt(f.a2),f.h1,f.h2);
      BOOST_CHECK_SMALL( arr::diff(f.F,F2,false,false), eps );

      // same with specialised version
      double bdvalue=arr::runif();
      pde::laplace(f.F,f.Uinput,f.a1,f.a2,f.h1,f.h2,bdvalue,bdtype);
      pde::grad(DX,DY,f.Uinput,sqrt(f.a1),sqrt(f.a2),f.h1,f.h2,bdvalue,bdtype);
      pde::div(F2,DX,DY,sqrt(f.a1),sqrt(f.a2),f.h1,f.h2);
      BOOST_CHECK_SMALL( arr::diff(f.F,F2,false,false), eps );
   }
   // same but with the boundary is contained in the array
   f.U_contains_boundary=true;
   f.resize(n1,n2);
   f.randomize();
   pde::laplace(f.F,f.Uinput,f.a1,f.a2,f.h1,f.h2);
   pde::grad(DX,DY,f.Uinput,sqrt(f.a1),sqrt(f.a2),f.h1,f.h2);
   pde::div(F2,DX,DY,sqrt(f.a1),sqrt(f.a2),f.h1,f.h2);
   BOOST_CHECK_SMALL( arr::diff(f.F,F2,false,false), eps );

}



// define U,  apply Laplace, and invert using poisolve
BOOST_AUTO_TEST_CASE( PoisonTest1 )
{
   const double eps=1e-9;
   size_t n1=140, n2=267;
   double error=0.0;
   MyFixture f;

   // version where U contains the boundary
   f.U_contains_boundary=true;
   f.resize(n1,n2);
   f.randomize();
   for(size_t i=pde::types::Dirichlet; i<=pde::types::Neumann; i++) {
      pde::types::boundary bdtype = static_cast<pde::types::boundary>(i);
      bool allow_shift = (bdtype==pde::types::Neumann) ? true : false;
      bool ignore_corners = f.U_contains_boundary;
      bool add_boundary = f.U_contains_boundary;

      pde::get_boundary(f.bd1a,f.bd1b,f.bd2a,f.bd2b,f.Uinput,f.h1,f.h2,bdtype);
      pde::laplace(f.F, f.Uinput, f.a1, f.a2, f.h1, f.h2);
      error=pde::poisolve(f.U, f.F, f.a1, f.a2, f.h1, f.h2,
                          f.bd1a, f.bd1b, f.bd2a, f.bd2b, bdtype, add_boundary);
      if(bdtype==pde::types::Neumann) {
         BOOST_CHECK_SMALL(error, eps);
      } else {
         BOOST_CHECK_EQUAL(error, 0.0);
      }
      BOOST_CHECK_SMALL(arr::diff(f.Uinput,f.U,allow_shift,ignore_corners), eps );
   }


   // version where U only contains inner points only and boundary is given
   // by boundary vectors
   f.U_contains_boundary=false;
   f.resize(n1,n2);
   f.randomize();
   for(size_t i=pde::types::Dirichlet; i<=pde::types::Neumann; i++) {
      pde::types::boundary bdtype = static_cast<pde::types::boundary>(i);

      bool allow_shift = (bdtype==pde::types::Neumann) ? true : false;
      bool ignore_corners = f.U_contains_boundary;
      bool add_boundary = f.U_contains_boundary;

      // version with boundary supplied by 4 vectors
      pde::laplace(f.F, f.Uinput, f.a1, f.a2, f.h1, f.h2,
                   f.bd1a, f.bd1b, f.bd2a, f.bd2b, bdtype);
      error=pde::poisolve(f.U, f.F, f.a1, f.a2, f.h1, f.h2,
                          f.bd1a, f.bd1b, f.bd2a, f.bd2b, bdtype, add_boundary);
      if(bdtype==pde::types::Neumann) {
         BOOST_CHECK_SMALL(error, eps);
      } else {
         BOOST_CHECK_EQUAL(error, 0.0);
      }
      BOOST_CHECK_SMALL(arr::diff(f.Uinput,f.U,allow_shift,ignore_corners), eps );


      // version with a constant boundary value
      double c=arr::runif();
      pde::laplace(f.F, f.Uinput, f.a1, f.a2, f.h1, f.h2, c, bdtype);
      pde::poisolve(f.U, f.F, f.a1, f.a2, f.h1, f.h2, c, bdtype, add_boundary);
      BOOST_CHECK_SMALL(arr::diff(f.Uinput,f.U,allow_shift,ignore_corners), eps );

   }
}


// define F,  solve U so that Laplace U = F, apply Laplace
BOOST_AUTO_TEST_CASE( PoisonTest2 )
{
   const double eps=1e-9;
   size_t n1=140, n2=267;
   double error_estim1=0.0;
   double error_estim2=0.0;
   double error_actual=0.0;
   MyFixture f;

   f.U_contains_boundary=false;
   f.resize(n1,n2);
   f.randomize();

   for(size_t i=pde::types::Dirichlet; i<=pde::types::Neumann; i++) {
      pde::types::boundary bdtype = static_cast<pde::types::boundary>(i);

      // with variable boundary values
      bool add_boundary=true;
      error_estim1=pde::poisolve(f.U, f.Finput, f.a1, f.a2, f.h1, f.h2,
                                 f.bd1a, f.bd1b, f.bd2a, f.bd2b, bdtype, add_boundary);
      pde::laplace(f.F, f.U, f.a1, f.a2, f.h1, f.h2);
      error_actual=arr::diff(f.Finput,f.F,false,false);
      BOOST_CHECK_SMALL(error_estim1-error_actual, eps);


      // with zero boundary value, for Neumann in general not solvable
      add_boundary=false;
      double bdvalue=0.0;
      error_estim1=pde::poisolve(f.U, f.Finput, f.a1, f.a2, f.h1, f.h2,
                                 bdvalue, bdtype, add_boundary);
      pde::laplace(f.F, f.U, f.a1, f.a2, f.h1, f.h2,bdvalue,bdtype);
      error_actual=arr::diff(f.Finput,f.F,false,false);
      if(bdtype==pde::types::Neumann) {
         // error estim for 0-Neumann boundary only
         error_estim2=pde::neumann_error(f.Finput);
         BOOST_CHECK_SMALL(error_estim2-error_actual, eps );
         BOOST_CHECK_SMALL(error_estim1-error_actual, eps );
      } else {
         BOOST_CHECK_EQUAL(error_estim1, 0.0);
         BOOST_CHECK_SMALL(error_actual, eps );
      }

      // with const boundary value
      // for Neumann we will ask for the only correct boundary value which
      // leads to a solution
      add_boundary=false;
      if(bdtype==pde::types::Neumann) {
         bdvalue=pde::neumann_compat(f.Finput,f.a1, f.a2, f.h1, f.h2);
      } else {
         bdvalue=arr::runif();
      }
      error_estim1=pde::poisolve(f.U, f.Finput, f.a1, f.a2, f.h1, f.h2,
                                 bdvalue, bdtype, add_boundary);
      pde::laplace(f.F, f.U, f.a1, f.a2, f.h1, f.h2,bdvalue,bdtype);
      error_actual=arr::diff(f.Finput,f.F,false,false);
      // error should now be zero also for Neumann
      BOOST_CHECK_SMALL(error_estim1, eps );
      BOOST_CHECK_SMALL(error_actual, eps );

   }




}
